"""FastAPI entrypoint for MIG-WeldSight AI backend.

Endpoints
---------
REST
  GET  /health
  GET  /api/profiles
  GET  /api/profiles/{material}/{t}
  POST /api/train
  POST /api/infer
  GET  /api/events

WebSocket
  /ws/stream  ESP32  -> backend
  /ws/live    browser <- backend
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .db import get_session, engine
from .models import Base, Profile, AnomalyEvent
from .features import extract, windowize
from .dynamic_threshold import DynamicThreshold
from .physics_classifier import classify
from .quality import compute_quality_index
from .training import train_baseline
from .analytics import project_and_cluster

app = FastAPI(title="MIG-WeldSight AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _create_tables():
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health(): return {"status": "ok", "ts": int(time.time())}


class ProfileIn(BaseModel):
    material: str
    thickness_mm: float
    good_welds: List[Dict[str, List[float]]]


def _profile_to_dict(p: Profile) -> dict:
    return {
        "material": p.material, "thickness_mm": float(p.thickness_mm),
        "learned_k": float(p.learned_k), "mean_score": float(p.mean_score),
        "std_score": float(p.std_score),
        "voltage_min": float(p.voltage_min) if p.voltage_min is not None else None,
        "voltage_max": float(p.voltage_max) if p.voltage_max is not None else None,
        "rms_min": float(p.rms_min) if p.rms_min is not None else None,
        "rms_max": float(p.rms_max) if p.rms_max is not None else None,
        "trained_windows": int(p.trained_windows),
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }


def _load_profile(material: str, thickness_mm: float):
    with get_session() as s:
        p = s.query(Profile).filter_by(material=material, thickness_mm=thickness_mm).first()
        return _profile_to_dict(p) if p else None


def _persist_event(distance, score, threshold, cls, quality_value, material, thickness_mm):
    with get_session() as s:
        s.add(AnomalyEvent(
            material=material, thickness_mm=thickness_mm, distance_mm=distance,
            anomaly_score=score, threshold=threshold, physics_label=cls["physics_label"],
            severity=cls["severity"], quality_index=quality_value, voltage_features=cls["features"],
        ))


@app.get("/api/profiles")
def list_profiles():
    with get_session() as s:
        return [_profile_to_dict(p) for p in s.query(Profile).all()]


@app.get("/api/profiles/{material}/{thickness_mm}")
def get_profile(material: str, thickness_mm: float):
    with get_session() as s:
        p = s.query(Profile).filter_by(material=material, thickness_mm=thickness_mm).first()
        if not p:
            raise HTTPException(404, "profile not found")
        return _profile_to_dict(p)


@app.post("/api/train")
def train(req: ProfileIn):
    result = train_baseline(req.good_welds, req.material, req.thickness_mm)
    with get_session() as s:
        p = s.query(Profile).filter_by(material=req.material, thickness_mm=req.thickness_mm).first()
        if not p:
            p = Profile(material=req.material, thickness_mm=req.thickness_mm,
                        learned_k=result["learned_k"], mean_score=result["mean_score"],
                        std_score=result["std_score"], voltage_min=result["voltage_min"],
                        voltage_max=result["voltage_max"], rms_min=result["rms_min"],
                        rms_max=result["rms_max"], trained_windows=result["trained_windows"])
            s.add(p)
        else:
            for k, v in result.items():
                if hasattr(p, k):
                    setattr(p, k, v)
    return result


class InferIn(BaseModel):
    material: str
    thickness_mm: float
    voltage: List[float]
    distance: List[float] | None = None


@app.post("/api/infer")
def infer(req: InferIn):
    profile = _load_profile(req.material, req.thickness_mm)
    k = profile["learned_k"] if profile else 3.0
    dt = DynamicThreshold(learned_k=k)
    distance = req.distance or list(range(len(req.voltage)))

    frames: List[dict] = []
    feature_rows: List[List[float]] = []
    for d_mid, feats in windowize(req.voltage, distance, size=64, step=32):
        score = feats.std_v + max(0.0, feats.crest_factor - 1.1) * 4.0
        t = dt.update(score)
        cls = classify(feats, score, t["threshold"], req.material, req.thickness_mm)
        quality = compute_quality_index(feats.std_v, feats.sc_count, feats.crest_factor, score, t["ewma"])
        frames.append({
            "distance_mm": d_mid, "anomaly_score": round(score, 3),
            "threshold": t["threshold"], "quality_index": quality["value"],
            "physics_label": cls["physics_label"], "severity": cls["severity"],
            "display_label": cls["display_label"], "possible_causes": cls["possible_causes"],
            "recommended_actions": cls["recommended_actions"], "voltage_features": feats.to_dict(),
        })
        feature_rows.append([feats.mean_v, feats.std_v, feats.rms_v, feats.sc_count, feats.crest_factor])
        if cls["severity"] != "NORMAL":
            _persist_event(d_mid, score, t["threshold"], cls, quality["value"], req.material, req.thickness_mm)
    return {"frames": frames, "cluster": project_and_cluster(feature_rows)}


@app.get("/api/events")
def events(limit: int = Query(200, le=2000)):
    with get_session() as s:
        rows = s.query(AnomalyEvent).order_by(AnomalyEvent.ts.desc()).limit(limit).all()
        return [{
            "ts": r.ts.isoformat() if r.ts else None, "material": r.material,
            "thickness_mm": float(r.thickness_mm), "distance_mm": float(r.distance_mm or 0),
            "anomaly_score": float(r.anomaly_score or 0), "threshold": float(r.threshold or 0),
            "physics_label": r.physics_label, "severity": r.severity,
            "quality_index": r.quality_index, "voltage_features": r.voltage_features,
        } for r in rows]


class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def add(self, ws: WebSocket):
        await ws.accept()
        async with self.lock:
            self.clients.add(ws)

    async def remove(self, ws: WebSocket):
        async with self.lock:
            self.clients.discard(ws)

    async def broadcast(self, frame: dict):
        async with self.lock:
            stale: list[WebSocket] = []
            for c in self.clients:
                try:
                    await c.send_text(json.dumps(frame))
                except Exception:
                    stale.append(c)
            for c in stale:
                self.clients.discard(c)


hub = Hub()


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await hub.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.remove(ws)


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    sample_buf: list[float] = []
    distance_buf: list[float] = []
    material = "mild_steel"
    thickness_mm = 6.0
    profile = _load_profile(material, thickness_mm)
    k = profile["learned_k"] if profile else 3.0
    dt = DynamicThreshold(learned_k=k)
    try:
        while True:
            raw = await ws.receive_text()
            msg: Dict[str, Any] = json.loads(raw)
            if "material" in msg or "thickness_mm" in msg:
                material = msg.get("material", material)
                thickness_mm = float(msg.get("thickness_mm", thickness_mm))
                profile = _load_profile(material, thickness_mm)
                k = profile["learned_k"] if profile else 3.0
                dt = DynamicThreshold(learned_k=k)
                continue

            voltage = float(msg.get("voltage", 0.0))
            distance = float(msg.get("distance_mm", 0.0))
            arc_on = bool(msg.get("arc_on", True))
            if not arc_on:
                continue
            sample_buf.append(voltage)
            distance_buf.append(distance)
            if len(sample_buf) >= 64:
                feats = extract(sample_buf)
                score = feats.std_v + max(0.0, feats.crest_factor - 1.1) * 4.0
                t = dt.update(score)
                cls = classify(feats, score, t["threshold"], material, thickness_mm)
                quality = compute_quality_index(feats.std_v, feats.sc_count, feats.crest_factor, score, t["ewma"])
                frame = {
                    "timestamp": int(time.time() * 1000),
                    "voltage": voltage, "distance_mm": distance_buf[-1],
                    "anomaly_score": round(score, 3), "threshold": t["threshold"],
                    "quality_index": quality["value"], "physics_label": cls["physics_label"],
                    "severity": cls["severity"], "display_label": cls["display_label"],
                    "material": material, "thickness_mm": thickness_mm,
                    "voltage_features": feats.to_dict(),
                    "possible_causes": cls["possible_causes"],
                    "recommended_actions": cls["recommended_actions"],
                }
                if cls["severity"] != "NORMAL":
                    _persist_event(distance_buf[-1], score, t["threshold"], cls, quality["value"], material, thickness_mm)
                await hub.broadcast(frame)
                sample_buf = sample_buf[32:]
                distance_buf = distance_buf[32:]
    except WebSocketDisconnect:
        return
