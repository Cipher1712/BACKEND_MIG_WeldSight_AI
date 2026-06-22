"""Production FastAPI entrypoint for WeldSight AI."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import math
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .analytics import project_and_cluster
from .db import engine, get_session, repair_sqlite_autoincrement_tables
from .features import WINDOW_SIZE, WINDOW_STRIDE, extract, windowize
from .inference import InferencePipeline
from .models import AnomalyEvent, Base, Profile
from .telemetry_state import telemetry_state
from .training import train_baseline

logger = logging.getLogger("weldsight")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
pipeline = InferencePipeline(os.getenv("MODEL_DIR", "models"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    repair_sqlite_autoincrement_tables()
    Base.metadata.create_all(bind=engine)
    logger.info("WeldSight started; model_ready=%s window=%s stride=%s", pipeline.ready, WINDOW_SIZE, WINDOW_STRIDE)
    yield


app = FastAPI(title="WeldSight AI", version="2.0.0", lifespan=lifespan)
origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=origins != ["*"],
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy", "timestamp": int(time.time()),
        **pipeline.health(),
        "window_size": WINDOW_SIZE, "stride": WINDOW_STRIDE,
        "sampling_rate_hz": 750,
    }


class ProfileIn(BaseModel):
    material: str
    thickness_mm: float = Field(gt=0, le=100)
    good_welds: list[dict[str, list[float]]]


class InferIn(BaseModel):
    material: str = "mild_steel"
    thickness_mm: float = Field(default=6.0, gt=0, le=100)
    voltage: list[float] = Field(min_length=WINDOW_SIZE, max_length=2_000_000)
    distance: list[float] | None = None


class TelemetryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voltage: list[float] = Field(min_length=1, max_length=20_000)
    distance_mm: float = 0.0
    arc_on: bool = True
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
    material: str = "mild_steel"
    thickness_mm: float = Field(default=6.0, gt=0, le=100)

    @field_validator("voltage")
    @classmethod
    def voltage_must_be_finite(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(value) for value in values):
            raise ValueError("voltage values must be finite numbers")
        return values

    @field_validator("distance_mm", "thickness_mm")
    @classmethod
    def scalar_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be a finite number")
        return value


def _profile_to_dict(profile: Profile) -> dict:
    return {
        "material": profile.material, "thickness_mm": float(profile.thickness_mm),
        "learned_k": float(profile.learned_k), "mean_score": float(profile.mean_score),
        "std_score": float(profile.std_score), "voltage_min": float(profile.voltage_min or 0),
        "voltage_max": float(profile.voltage_max or 0), "rms_min": float(profile.rms_min or 0),
        "rms_max": float(profile.rms_max or 0), "trained_windows": int(profile.trained_windows),
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }


def _persist_event(distance: float, result: dict, material: str, thickness_mm: float) -> None:
    with get_session() as session:
        session.add(AnomalyEvent(
            material=material, thickness_mm=thickness_mm, distance_mm=distance,
            anomaly_score=result["anomaly_score"], threshold=result.get("anomaly_threshold", 0.60),
            physics_label=result["physics_label"], severity=result["severity"],
            quality_index=result["quality_index"], voltage_features=result["voltage_features"],
        ))


def _event_row_to_dict(row: AnomalyEvent) -> dict:
    return {
        "timestamp": row.ts.isoformat() if row.ts else None, "material": row.material,
        "thickness_mm": float(row.thickness_mm), "distance_mm": float(row.distance_mm or 0),
        "anomaly_score": float(row.anomaly_score or 0), "physics_label": row.physics_label,
        "severity": row.severity, "quality_index": row.quality_index,
        "voltage_features": row.voltage_features,
    }


def _build_live_frame(
    *,
    timestamp: int,
    voltage: float,
    distance_mm: float,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "voltage": voltage,
        "distance_mm": distance_mm,
        "quality_score": result["quality_score"],
        "status": result["status"],
        "diagnosis": result["diagnosis"],
        "top_contributors": result["top_contributors"],
        "quality_index": result["quality_index"],
        "quality_category": result["quality_category"],
        "severity": result["severity"],
        "anomaly_score": result["anomaly_score"],
        "anomaly_threshold": result["anomaly_threshold"],
        "anomaly_detected": result["anomaly_detected"],
        "anomaly_stages": result["anomaly_stages"],
        "physics_label": result["physics_label"],
        "ml_label": result["ml_label"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "top_features": result["top_features"],
        "recommendation": result["recommendation"],
        "explanation": result["explanation"],
        "stability_score": result["stability_score"],
        "voltage_features": result["voltage_features"],
        "model_ready": result["model_ready"],
    }


async def _process_telemetry_packet(packet: dict[str, Any], material: str, thickness_mm: float) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    windows = telemetry_state.append_packet(packet)
    for window, distance_mm in windows:
        features = extract(window)
        result = pipeline.predict_features(features, material, thickness_mm)
        frame = _build_live_frame(
            timestamp=int(packet["timestamp"]),
            voltage=float(window[-1]),
            distance_mm=distance_mm,
            result=result,
        )
        if result["anomaly_detected"]:
            await asyncio.to_thread(_persist_event, distance_mm, result, material, thickness_mm)
        telemetry_state.record_frame(frame)
        await hub.broadcast(frame)
        frames.append(frame)
    return frames


@app.get("/api/profiles")
def list_profiles() -> list[dict]:
    with get_session() as session:
        return [_profile_to_dict(row) for row in session.query(Profile).all()]


@app.get("/api/profiles/{material}/{thickness_mm}")
def get_profile(material: str, thickness_mm: float) -> dict:
    with get_session() as session:
        profile = session.query(Profile).filter_by(material=material, thickness_mm=thickness_mm).first()
        if profile is None:
            raise HTTPException(404, "profile not found")
        return _profile_to_dict(profile)


@app.post("/api/train")
def train_profile(request: ProfileIn) -> dict:
    """Retained lightweight profile calibration; offline scripts train ML artifacts."""
    result = train_baseline(request.good_welds, request.material, request.thickness_mm)
    with get_session() as session:
        profile = session.query(Profile).filter_by(
            material=request.material, thickness_mm=request.thickness_mm
        ).first()
        if profile is None:
            profile = Profile(**{key: value for key, value in result.items() if key != "material"},
                              material=request.material)
            session.add(profile)
        else:
            for key, value in result.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
    return result


@app.post("/api/infer")
def infer(request: InferIn) -> dict:
    if request.distance is not None and len(request.distance) != len(request.voltage):
        raise HTTPException(422, "distance and voltage arrays must have equal length")
    frames, feature_rows = [], []
    for distance, features in windowize(request.voltage, request.distance):
        result = pipeline.predict_features(features, request.material, request.thickness_mm)
        frame = {"timestamp": int(time.time() * 1000), "voltage": features.mean_v,
                 "distance_mm": distance, **result}
        frames.append(frame)
        feature_rows.append(features.to_vector().tolist())
        if result["anomaly_detected"]:
            _persist_event(distance, result, request.material, request.thickness_mm)
    return {"frames": frames, "cluster": project_and_cluster(feature_rows), "model_ready": pipeline.ready}


@app.get("/api/events")
def events(limit: int = Query(200, ge=1, le=2000)) -> list[dict]:
    with get_session() as session:
        rows = session.query(AnomalyEvent).order_by(AnomalyEvent.ts.desc()).limit(limit).all()
        return [_event_row_to_dict(row) for row in rows]


@app.post("/telemetry")
async def ingest_telemetry(request: TelemetryIn) -> dict:
    packet = request.model_dump()
    frames = await _process_telemetry_packet(packet, request.material, request.thickness_mm)
    return {
        "status": "accepted",
        "timestamp": request.timestamp,
        "samples_received": len(request.voltage),
        "frames_processed": len(frames),
        "model_ready": pipeline.ready,
    }


@app.get("/telemetry/latest")
def latest_telemetry() -> dict:
    return telemetry_state.latest_telemetry()


@app.get("/metrics/latest")
def latest_metrics() -> dict:
    return telemetry_state.latest_metrics()


@app.get("/events/latest")
def latest_events(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    state_events = telemetry_state.latest_events()
    if state_events:
        return state_events[:limit]
    with get_session() as session:
        rows = session.query(AnomalyEvent).order_by(AnomalyEvent.ts.desc()).limit(limit).all()
        return [_event_row_to_dict(row) for row in rows]


class Hub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def add(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self.lock:
            self.clients.add(websocket)

    async def remove(self, websocket: WebSocket) -> None:
        async with self.lock:
            self.clients.discard(websocket)

    async def broadcast(self, frame: dict) -> None:
        payload = json.dumps(frame, separators=(",", ":"), allow_nan=False)
        async with self.lock:
            clients = tuple(self.clients)
        results = await asyncio.gather(*(client.send_text(payload) for client in clients), return_exceptions=True)
        stale = [client for client, result in zip(clients, results) if isinstance(result, Exception)]
        if stale:
            async with self.lock:
                for client in stale:
                    self.clients.discard(client)


hub = Hub()


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    # Deprecated: retained temporarily for frontend rollback during HTTP polling migration.
    await hub.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.remove(websocket)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    # Deprecated: retained temporarily for ESP32 rollback during HTTP telemetry migration.
    await websocket.accept()
    material, thickness_mm = "mild_steel", 6.0
    try:
        while True:
            message: dict[str, Any] = json.loads(await websocket.receive_text())
            if "material" in message or "thickness_mm" in message:
                material = str(message.get("material", material))
                thickness_mm = float(message.get("thickness_mm", thickness_mm))
                continue
            if not bool(message.get("arc_on", True)):
                await _process_telemetry_packet({
                    "voltage": [0.0],
                    "distance_mm": float(message.get("distance_mm", 0.0)),
                    "arc_on": False,
                    "timestamp": int(message.get("timestamp", time.time() * 1000)),
                    "material": material,
                    "thickness_mm": thickness_mm,
                }, material, thickness_mm)
                continue
            values = message.get("voltage")
            samples = values if isinstance(values, list) else [values]
            if not samples or samples[0] is None:
                continue
            await _process_telemetry_packet({
                "voltage": [float(value) for value in samples],
                "distance_mm": float(message.get("distance_mm", 0.0)),
                "arc_on": True,
                "timestamp": int(message.get("timestamp", time.time() * 1000)),
                "material": material,
                "thickness_mm": thickness_mm,
            }, material, thickness_mm)
    except (WebSocketDisconnect, ValueError, json.JSONDecodeError):
        return
    except Exception:
        logger.exception("stream processing failed")
        await websocket.close(code=1011)
