"""Artifact-backed real-time inference pipeline with safe physics fallback."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .anomaly import AnomalyDetector
from .explainability import Explainer, explanation_text
from .features import WindowFeatures, extract
from .physics import assess
from .quality import compute_quality_index


class InferencePipeline:
    def __init__(self, model_dir: str | os.PathLike = "models"):
        path = Path(model_dir)
        self.scaler = joblib.load(path / "scaler.pkl") if (path / "scaler.pkl").exists() else None
        self.classifier = joblib.load(path / "classifier.pkl") if (path / "classifier.pkl").exists() else None
        isolation = joblib.load(path / "isolation_forest.pkl") if (path / "isolation_forest.pkl").exists() else None
        thresholds = json.loads((path / "thresholds.json").read_text()) if (path / "thresholds.json").exists() else {}
        if (path / "vae.pt").exists():
            from .vae import load_vae
            vae = load_vae(str(path / "vae.pt"))
        else:
            vae = None
        self.detector = AnomalyDetector(thresholds, isolation, vae)
        self.explainer = Explainer(self.classifier, WindowFeatures.names()) if self.classifier is not None else None
        self.ready = all(item is not None for item in (self.scaler, self.classifier, vae))

    def predict_features(
        self, features: WindowFeatures, material: str = "mild_steel", thickness_mm: float = 6.0
    ) -> dict[str, Any]:
        physics = assess(features, material, thickness_mm)
        raw = features.to_vector().reshape(1, -1)
        scaled = self.scaler.transform(raw) if self.scaler is not None else raw
        anomaly = self.detector.score(scaled[0], physics.score)

        if self.classifier is not None:
            probabilities = self.classifier.predict_proba(scaled)[0]
            index = int(np.argmax(probabilities))
            ml_label = str(self.classifier.classes_[index])
            confidence = float(probabilities[index])
            top_features = self.explainer.explain(scaled, ml_label) if self.explainer else []
        else:
            ml_label = physics.label if not anomaly.is_anomaly else (
                physics.label if physics.label != "stable_arc" else "unknown_anomaly"
            )
            confidence = max(0.50, anomaly.score if anomaly.is_anomaly else 1.0 - anomaly.score)
            ranked = sorted(features.to_dict().items(), key=lambda item: abs(float(item[1])), reverse=True)[:3]
            top_features = [{"feature": k, "value": round(float(v), 5), "impact": 0.0} for k, v in ranked]

        quality = compute_quality_index(
            physics.score, anomaly.score, confidence, physics.stability_score, ml_label == "stable_arc"
        )
        severity = (
            "CRITICAL" if quality["value"] < 30 else "POOR" if quality["value"] < 55 else
            "WARNING" if quality["value"] < 75 else "NORMAL"
        )
        return {
            "quality_index": quality["value"], "quality_category": quality["band"],
            "severity": severity, "anomaly_score": anomaly.score,
            "anomaly_threshold": anomaly.threshold,
            "anomaly_detected": anomaly.is_anomaly, "anomaly_stages": {
                "ewma": anomaly.stage1_score, "isolation_forest": anomaly.stage2_score,
                "physics_vae": anomaly.stage3_score,
            },
            "physics_label": physics.label, "physics_score": physics.score,
            "stability_score": physics.stability_score, "ml_label": ml_label,
            "prediction": ml_label, "confidence": round(confidence, 5),
            "top_features": top_features, "explanation": explanation_text(ml_label, top_features),
            "recommendation": physics.recommendation, "voltage_features": features.to_dict(),
            "model_ready": self.ready,
        }

    def predict(self, voltage: list[float], material: str = "mild_steel", thickness_mm: float = 6.0) -> dict:
        return self.predict_features(extract(voltage), material, thickness_mm)
