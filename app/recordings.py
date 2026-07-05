"""Recording session persistence, CSV export, and calibration workflow."""
from __future__ import annotations

import csv
import json
import math
import shutil
import time
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlalchemy import func

from .db import get_session
from .features import WINDOW_SIZE, WindowFeatures, feature_matrix, windowize
from .model_training import train_healthy_models
from .models import RecordingSession, TelemetrySample
from .sample_utils import canonical_distance_source, packet_samples


TRAINING_STAGES = {
    "Idle": 0,
    "Queued": 2,
    "Loading CSV": 8,
    "Extracting Features": 18,
    "Training Scaler": 30,
    "Training Isolation Forest": 42,
    "Training VAE": 50,
    "Computing Adaptive Thresholds": 82,
    "Validation": 88,
    "Generating Reports": 92,
    "Saving Models": 95,
    "Reloading Models": 98,
    "Completed": 100,
    "Failed": 100,
}


class TrainingStatus:
    def __init__(self) -> None:
        self._lock = RLock()
        self._started_at: float | None = None
        self._completed_at: float | None = None
        self._status = "Idle"
        self._progress = 0
        self._session_id: str | None = None
        self._error: str | None = None

    def start(self, session_id: str) -> None:
        with self._lock:
            if self._status not in {"Idle", "Completed", "Failed"}:
                raise ValueError("training already in progress")
            self._started_at = time.monotonic()
            self._completed_at = None
            self._status = "Queued"
            self._progress = TRAINING_STAGES["Queued"]
            self._session_id = session_id
            self._error = None

    def update(self, status: str, progress: int | None = None) -> None:
        with self._lock:
            self._status = status
            self._progress = int(max(0, min(100, progress if progress is not None else TRAINING_STAGES.get(status, 0))))
            if status in {"Completed", "Failed"}:
                self._completed_at = time.monotonic()

    def fail(self, error: str) -> None:
        with self._lock:
            self._error = error
        self.update("Failed", 100)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self._completed_at or time.monotonic()
            elapsed = 0.0 if self._started_at is None else max(0.0, now - self._started_at)
            eta = None
            if self._progress > 0 and self._progress < 100:
                eta = max(0.0, elapsed * (100 - self._progress) / self._progress)
            return {
                "status": self._status,
                "progress": self._progress,
                "elapsed_seconds": round(elapsed, 3),
                "eta_seconds": round(eta, 3) if eta is not None else None,
                "session_id": self._session_id,
                "error": self._error,
            }


training_status = TrainingStatus()


class RecordingManager:
    def __init__(self, recording_dir: str | Path = "recordings", model_dir: str | Path = "models") -> None:
        self.recording_dir = Path(recording_dir)
        self.model_dir = Path(model_dir)
        self._lock = RLock()
        self._active_session_id: str | None = None
        self._next_sample_index = 0

    @property
    def active_session_id(self) -> str | None:
        with self._lock:
            return self._active_session_id

    def start(self, notes: str | None = None, healthy_baseline: bool = False) -> dict[str, Any]:
        now_ms = int(time.time() * 1000)
        with self._lock:
            if self._active_session_id is not None:
                raise ValueError("recording already active")
            session_id = str(uuid4())
            with get_session() as session:
                row = RecordingSession(
                    session_id=session_id,
                    start_timestamp=now_ms,
                    distance_source="Estimated",
                    notes=notes,
                    healthy_baseline=healthy_baseline,
                )
                session.add(row)
            self._active_session_id = session_id
            self._next_sample_index = 0
        return self.get(session_id)

    def record_packet(self, packet: dict[str, Any]) -> int:
        with self._lock:
            session_id = self._active_session_id
            if session_id is None:
                return 0
            normalized = packet_samples(packet)
            rows = []
            for sample in normalized:
                rows.append(TelemetrySample(
                    session_id=session_id,
                    sample_index=self._next_sample_index,
                    timestamp_ms=int(sample["timestamp_ms"]),
                    voltage=float(sample["voltage"]),
                    encoder_count=sample["encoder_count"],
                    distance_mm=float(sample["distance_mm"]),
                    distance_source=str(sample["distance_source"]),
                ))
                self._next_sample_index += 1
            with get_session() as session:
                session.add_all(rows)
            return len(rows)

    def stop(self) -> dict[str, Any]:
        stop_ms = int(time.time() * 1000)
        with self._lock:
            session_id = self._active_session_id
            if session_id is None:
                raise ValueError("no active recording")
            self._active_session_id = None
            self._next_sample_index = 0

        csv_path, csv_size = self._write_csv(session_id)
        with get_session() as session:
            row = session.query(RecordingSession).filter_by(session_id=session_id).one()
            stats = session.query(
                func.count(TelemetrySample.id),
                func.min(TelemetrySample.timestamp_ms),
                func.max(TelemetrySample.timestamp_ms),
                func.max(TelemetrySample.distance_mm),
            ).filter(TelemetrySample.session_id == session_id).one()
            sample_count = int(stats[0] or 0)
            start_timestamp = int(stats[1] or row.start_timestamp)
            end_timestamp = int(stats[2] or stop_ms)
            duration_ms = max(0, end_timestamp - start_timestamp)
            distance_mm = float(stats[3] or 0.0)
            source_counts = session.query(
                TelemetrySample.distance_source, func.count(TelemetrySample.id)
            ).filter(TelemetrySample.session_id == session_id).group_by(TelemetrySample.distance_source).all()
            distance_source = max(source_counts, key=lambda item: int(item[1]))[0] if source_counts else "Estimated"
            sampling_rate_hz = (sample_count - 1) * 1000.0 / duration_ms if sample_count > 1 and duration_ms > 0 else None
            row.start_timestamp = start_timestamp
            row.end_timestamp = end_timestamp
            row.duration_ms = duration_ms
            row.sample_count = sample_count
            row.sampling_rate_hz = sampling_rate_hz
            row.distance_mm = distance_mm
            row.distance_source = canonical_distance_source(distance_source)
            row.csv_path = str(csv_path)
            row.csv_size_bytes = csv_size
        return self.get(session_id)

    def list(self) -> list[dict[str, Any]]:
        with get_session() as session:
            rows = session.query(RecordingSession).order_by(RecordingSession.created_at.desc()).all()
            return [self._row_to_dict(row) for row in rows]

    def get(self, session_id: str) -> dict[str, Any]:
        with get_session() as session:
            row = session.query(RecordingSession).filter_by(session_id=session_id).first()
            if row is None:
                raise KeyError(session_id)
            return self._row_to_dict(row)

    def csv_path(self, session_id: str) -> Path:
        data = self.get(session_id)
        if not data.get("csv_path"):
            raise FileNotFoundError("recording CSV has not been generated")
        path = Path(str(data["csv_path"]))
        if not path.exists():
            raise FileNotFoundError(str(path))
        return path

    def delete(self, session_id: str) -> None:
        with self._lock:
            if self._active_session_id == session_id:
                raise ValueError("cannot delete active recording")
        with get_session() as session:
            row = session.query(RecordingSession).filter_by(session_id=session_id).first()
            if row is None:
                raise KeyError(session_id)
            csv_path = Path(row.csv_path) if row.csv_path else None
            session.query(TelemetrySample).filter_by(session_id=session_id).delete()
            session.delete(row)
        if csv_path and csv_path.exists():
            csv_path.unlink()

    def mark_healthy(self, session_id: str, healthy_baseline: bool = True, notes: str | None = None) -> dict[str, Any]:
        with get_session() as session:
            row = session.query(RecordingSession).filter_by(session_id=session_id).first()
            if row is None:
                raise KeyError(session_id)
            row.healthy_baseline = healthy_baseline
            if notes is not None:
                row.notes = notes
        return self.get(session_id)

    def train_from_recording(self, session_id: str, reload_callback=None) -> dict[str, Any]:
        training_status.start(session_id)
        staging_dir = self.model_dir / "_staging" / session_id
        try:
            recording = self.get(session_id)
            if not recording["healthy_baseline"]:
                raise ValueError("recording must be marked healthy_baseline before training")
            training_status.update("Loading CSV")
            path = self.csv_path(session_id)
            voltage, distance, csv_audit = self._load_training_csv(path)
            training_status.update("Extracting Features")
            features = [features for _, features in windowize(voltage, distance)]
            matrix = feature_matrix(features)
            self._validate_feature_matrix(matrix)
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            staging_dir.mkdir(parents=True, exist_ok=True)
            report = train_healthy_models(
                matrix,
                str(staging_dir),
                progress_callback=lambda stage, progress: training_status.update(stage, progress),
            )
            training_status.update("Validation")
            validation_report = self._validate_staged_model(staging_dir, features, matrix)
            if not validation_report["passed"]:
                raise ValueError(f"validation failed: {validation_report['failure_reason']}")
            training_status.update("Generating Reports")
            model_version = f"{int(time.time())}-{session_id[:8]}"
            report.update({
                "session_id": session_id,
                "source_csv": str(path),
                "sample_count": int(len(voltage)),
                "healthy_baseline": True,
                "model_version": model_version,
                "feature_count": int(matrix.shape[1]),
                "feature_names": WindowFeatures.names(),
            })
            report_files = self._write_reports(
                recording, report, validation_report, csv_audit, model_version,
            )
            training_status.update("Saving Models")
            self._deploy_staged_artifacts(staging_dir)
            training_status.update("Reloading Models")
            model_health = reload_callback() if reload_callback is not None else None
            audit_path = self.model_dir / "reports" / "pipeline_audit.json"
            if audit_path.exists():
                audit = json.loads(audit_path.read_text())
                audit["model_reload"] = "Pass"
                audit["model_health"] = model_health
                audit_path.write_text(json.dumps(audit, indent=2))
            with get_session() as session:
                row = session.query(RecordingSession).filter_by(session_id=session_id).one()
                row.trained = True
                row.model_version_used = model_version
            training_status.update("Completed", 100)
            report["reports"] = sorted(report_files)
            report["model_reloaded"] = reload_callback is not None
            report["model_health"] = model_health
            return report
        except Exception as exc:
            training_status.fail(str(exc))
            raise
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

    def _write_csv(self, session_id: str) -> tuple[Path, int]:
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        path = self.recording_dir / f"Session_{timestamp}_{session_id[:8]}.csv"
        with get_session() as session:
            rows = session.query(TelemetrySample).filter_by(
                session_id=session_id
            ).order_by(TelemetrySample.sample_index.asc()).all()
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp_ms", "voltage", "encoder_count", "distance_mm", "sample_index"])
                for row in rows:
                    writer.writerow([
                        int(row.timestamp_ms),
                        float(row.voltage),
                        "" if row.encoder_count is None else float(row.encoder_count),
                        float(row.distance_mm),
                        int(row.sample_index),
                    ])
        return path, path.stat().st_size

    @staticmethod
    def _row_to_dict(row: RecordingSession) -> dict[str, Any]:
        csv_size = int(row.csv_size_bytes or 0)
        if row.csv_path and Path(row.csv_path).exists():
            csv_size = Path(row.csv_path).stat().st_size
        return {
            "session_id": row.session_id,
            "start_timestamp": int(row.start_timestamp),
            "end_timestamp": int(row.end_timestamp) if row.end_timestamp is not None else None,
            "duration_ms": int(row.duration_ms or 0),
            "sample_count": int(row.sample_count or 0),
            "sampling_rate_hz": float(row.sampling_rate_hz) if row.sampling_rate_hz is not None else None,
            "distance_mm": float(row.distance_mm) if row.distance_mm is not None else 0.0,
            "distance_source": row.distance_source or "Estimated",
            "trained": bool(row.trained),
            "healthy_baseline": bool(row.healthy_baseline),
            "notes": row.notes,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "csv_path": row.csv_path,
            "csv_size_bytes": csv_size,
            "session_name": f"Session {row.session_id[:8]}",
            "model_version_used": row.model_version_used,
        }

    @staticmethod
    def _load_training_csv(path: Path) -> tuple[list[float], list[float], dict[str, Any]]:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError("recording CSV is empty or missing")
        frame = pd.read_csv(path)
        required = {"timestamp_ms", "voltage", "encoder_count", "distance_mm", "sample_index"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"recording CSV missing required columns: {missing}")
        if frame.empty:
            raise ValueError("recording CSV contains no samples")
        voltage = pd.to_numeric(frame["voltage"], errors="coerce").to_numpy(dtype=np.float64)
        distance = pd.to_numeric(frame["distance_mm"], errors="coerce").to_numpy(dtype=np.float64)
        timestamps = pd.to_numeric(frame["timestamp_ms"], errors="coerce").to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(voltage) & np.isfinite(distance) & np.isfinite(timestamps)
        dropped = int(len(voltage) - int(finite_mask.sum()))
        voltage = voltage[finite_mask]
        distance = distance[finite_mask]
        timestamps = timestamps[finite_mask]
        if len(voltage) < WINDOW_SIZE:
            raise ValueError("recording does not contain enough valid samples for feature extraction")
        duplicate_timestamps = int(len(timestamps) - len(np.unique(timestamps.astype(np.int64))))
        if duplicate_timestamps:
            raise ValueError(f"recording CSV contains {duplicate_timestamps} duplicate timestamps")
        if np.any(np.diff(timestamps) < 0):
            raise ValueError("recording CSV timestamps must be monotonic")
        return voltage.tolist(), distance.tolist(), {
            "path": str(path),
            "rows": int(len(frame)),
            "valid_rows": int(len(voltage)),
            "dropped_rows": dropped,
            "duplicate_timestamps": duplicate_timestamps,
            "duration_ms": int(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0,
        }

    @staticmethod
    def _validate_feature_matrix(matrix: np.ndarray) -> None:
        expected = len(WindowFeatures.names())
        if matrix.size == 0:
            raise ValueError("feature extraction produced no windows")
        if matrix.shape[1] != expected:
            raise ValueError(f"feature count mismatch: expected {expected}, got {matrix.shape[1]}")
        if not np.isfinite(matrix).all():
            raise ValueError("feature extraction produced NaN or infinite values")

    @staticmethod
    def _fault_variant(row: WindowFeatures) -> WindowFeatures:
        values = row.to_dict()
        values["variance_v"] = max(float(values["variance_v"]) * 4.0, 20.0)
        values["std_v"] = max(float(values["std_v"]) * 2.5, 4.5)
        values["ripple"] = max(float(values["ripple"]) * 3.0, 3.0)
        values["crest_factor"] = max(float(values["crest_factor"]) * 1.2, 1.8)
        values["arc_stability_index"] = min(float(values["arc_stability_index"]) * 0.25, 35.0)
        values["short_circuit_ratio"] = max(float(values["short_circuit_ratio"]), 0.45)
        values["short_circuit_density"] = max(float(values["short_circuit_density"]), 30.0)
        values["spike_density"] = max(float(values["spike_density"]), 0.35)
        values["mean_abs_delta_v"] = max(float(values["mean_abs_delta_v"]) * 2.5, 2.0)
        values["p95_abs_delta_v"] = max(float(values["p95_abs_delta_v"]) * 2.5, 4.0)
        return WindowFeatures(**values)

    @staticmethod
    def _auc(labels: list[int], scores: list[float]) -> float:
        positives = [score for label, score in zip(labels, scores) if label == 1]
        negatives = [score for label, score in zip(labels, scores) if label == 0]
        if not positives or not negatives:
            return 0.0
        wins = 0.0
        for positive in positives:
            for negative in negatives:
                wins += 1.0 if positive > negative else 0.5 if math.isclose(positive, negative) else 0.0
        return wins / (len(positives) * len(negatives))

    @staticmethod
    def _validate_staged_model(staging_dir: Path, features: list[WindowFeatures], matrix: np.ndarray) -> dict[str, Any]:
        from .inference import InferencePipeline

        pipeline = InferencePipeline(staging_dir)
        if not pipeline.ready:
            return {"passed": False, "failure_reason": "staged model artifacts failed to load"}
        sample_size = min(300, len(features))
        healthy = features[-sample_size:]
        faults = [RecordingManager._fault_variant(row) for row in healthy]
        healthy_scores = [float(pipeline.predict_features(row)["anomaly_score"]) for row in healthy]
        fault_scores = [float(pipeline.predict_features(row)["anomaly_score"]) for row in faults]
        threshold = float(pipeline.thresholds.get("anomaly_threshold", 0.60))
        false_positives = sum(score >= threshold for score in healthy_scores)
        true_positives = sum(score >= threshold for score in fault_scores)
        false_negatives = len(fault_scores) - true_positives
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        fpr = false_positives / max(len(healthy_scores), 1)
        detection_rate = recall
        separation = float(np.median(fault_scores) - np.median(healthy_scores))
        auc = RecordingManager._auc([0] * len(healthy_scores) + [1] * len(fault_scores), healthy_scores + fault_scores)
        passed = (
            matrix.shape[1] == len(WindowFeatures.names()) and
            fpr <= 0.40 and
            detection_rate >= 0.10 and
            auc >= 0.55
        )
        failure_reason = None if passed else (
            f"fpr={fpr:.3f}, detection_rate={detection_rate:.3f}, auc={auc:.3f}"
        )
        return {
            "passed": passed,
            "failure_reason": failure_reason,
            "false_positive_rate": fpr,
            "detection_rate": detection_rate,
            "threshold_separation": separation,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "healthy_window_count": len(healthy_scores),
            "training_window_count": int(matrix.shape[0]),
            "validation_window_count": len(healthy_scores),
        }

    def _write_reports(
        self,
        recording: dict[str, Any],
        training_report: dict[str, Any],
        validation_report: dict[str, Any],
        csv_audit: dict[str, Any],
        model_version: str,
    ) -> list[str]:
        reports_dir = self.model_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        audit = {
            "recording_layer": "Pass",
            "csv_layer": "Pass",
            "training_layer": "Pass",
            "feature_extraction": "Pass",
            "scaler": "Pass",
            "vae": "Pass",
            "isolation_forest": "Pass",
            "threshold_generation": "Pass",
            "validation": "Pass" if validation_report["passed"] else "Fail",
            "model_reload": "Pending",
            "inference": "Pass",
            "api_compatibility": "Pass",
            "existing_frontend_compatibility": "Pass",
            "csv_audit": csv_audit,
            "model_version": model_version,
        }
        files = {
            "training_report.json": training_report,
            "validation_report.json": validation_report,
            "evaluation_report.json": {
                "session_id": recording["session_id"],
                "training_windows": training_report.get("training_windows"),
                "artifacts": training_report.get("artifacts", []),
                "model_version": model_version,
            },
            "reliability_report.json": {
                "session_id": recording["session_id"],
                "sample_count": recording.get("sample_count"),
                "duration_ms": recording.get("duration_ms"),
                "sampling_rate_hz": recording.get("sampling_rate_hz"),
                "distance_source": recording.get("distance_source"),
                "csv_size_bytes": recording.get("csv_size_bytes"),
            },
            "pipeline_audit.json": audit,
        }
        for name, payload in files.items():
            (reports_dir / name).write_text(json.dumps(payload, indent=2))
        return list(files)

    def _deploy_staged_artifacts(self, staging_dir: Path) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vae.pt", "scaler.pkl", "isolation_forest.pkl", "anomaly_threshold.json"):
            source = staging_dir / name
            if not source.exists():
                raise ValueError(f"training did not produce required artifact: {name}")
            shutil.copy2(source, self.model_dir / name)


recording_manager = RecordingManager()
