"""Artifact-backed healthy-only welding inference."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .anomaly import AnomalyDetector
from .features import WindowFeatures, extract
from .physics import assess
from .quality import compute_quality_index


class InferencePipeline:
    def __init__(self, model_dir: str | os.PathLike = "models"):
        path = Path(model_dir)
        threshold_path = path / "anomaly_threshold.json"
        self.scaler = joblib.load(path / "scaler.pkl") if (path / "scaler.pkl").exists() else None
        self.isolation_forest = (
            joblib.load(path / "isolation_forest.pkl")
            if (path / "isolation_forest.pkl").exists() else None
        )
        self.thresholds = json.loads(threshold_path.read_text()) if threshold_path.exists() else {}
        self.vae = None
        if (path / "vae.pt").exists():
            from .vae import load_vae
            self.vae = load_vae(str(path / "vae.pt"))
        self.detector = AnomalyDetector(self.thresholds, self.isolation_forest, self.vae)
        self.scaler_loaded = self.scaler is not None
        self.vae_loaded = self.vae is not None
        self.threshold_loaded = bool(self.thresholds)
        self.isolation_forest_loaded = self.isolation_forest is not None
        self.ready = all((
            self.scaler_loaded, self.vae_loaded, self.threshold_loaded,
            self.isolation_forest_loaded,
        ))

    def health(self) -> dict[str, bool]:
        return {
            "model_ready": self.ready, "vae_loaded": self.vae_loaded,
            "scaler_loaded": self.scaler_loaded,
            "threshold_loaded": self.threshold_loaded,
            "isolation_forest_loaded": self.isolation_forest_loaded,
        }

    def _contributors(self, scaled: np.ndarray) -> list[dict[str, float | str]]:
        if self.vae is not None:
            importance = self.vae.reconstruction_contributions(scaled)
            method = "vae_reconstruction"
        else:
            importance = np.abs(scaled[0])
            method = "standardized_deviation"
        indices = np.argsort(importance)[-5:][::-1]
        return [{
            "feature": WindowFeatures.names()[index],
            "importance": round(float(importance[index]), 6),
            "method": method,
        } for index in indices]

    def predict_features(self, features: WindowFeatures, *_: Any, **__: Any) -> dict[str, Any]:
        reference = self.thresholds.get("feature_reference", {})
        physics = assess(features, reference)
        raw = features.to_vector().reshape(1, -1)
        scaled = self.scaler.transform(raw) if self.scaler is not None else raw
        anomaly = self.detector.score(scaled[0], physics.score)
        variance_q99 = float(reference.get("variance_v", {}).get("q99", max(features.variance_v, 1.0)))
        variance_health = 1.0 - min(1.0, features.variance_v / max(variance_q99, 1e-9))
        quality = compute_quality_index(
            physics.stability_score, variance_health, anomaly.vae_score,
            anomaly.isolation_score, physics.score,
        )
        status = (
            "Critical Arc" if quality["value"] < 30 else
            "Unstable Arc" if anomaly.is_anomaly or physics.label != "healthy_arc" else
            "Healthy Arc"
        )
        contributors = self._contributors(scaled)
        severity = (
            "CRITICAL" if quality["value"] < 30 else "POOR" if quality["value"] < 55 else
            "WARNING" if quality["value"] < 75 else "NORMAL"
        )
        return {
            # New healthy-only contract.
            "quality_score": quality["value"], "anomaly_score": anomaly.score,
            "status": status, "diagnosis": physics.diagnosis,
            "top_contributors": contributors,
            # Existing frontend-compatible fields.
            "quality_index": quality["value"], "quality_category": quality["band"],
            "severity": severity, "anomaly_threshold": anomaly.threshold,
            "anomaly_detected": anomaly.is_anomaly,
            "anomaly_stages": {
                "ewma": anomaly.ewma_score,
                "isolation_forest": anomaly.isolation_score,
                "physics_vae": anomaly.vae_score,
            },
            "physics_label": physics.label, "ml_label": physics.label,
            "prediction": physics.label,
            "confidence": round(1.0 - min(anomaly.score, 1.0), 5),
            "top_features": contributors, "explanation": physics.diagnosis,
            "recommendation": physics.recommendation,
            "stability_score": round(physics.stability_score, 4),
            "voltage_features": features.to_dict(), "model_ready": self.ready,
        }

    def predict(self, voltage: list[float], *args: Any, **kwargs: Any) -> dict:
        return self.predict_features(extract(voltage), *args, **kwargs)
