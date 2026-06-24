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


def _confidence(label: str, rule_strength: float, strong_rule_count: int, anomaly_score: float, threshold: float) -> float:
    anomaly_severity = float(np.clip(anomaly_score / max(threshold, 1e-9), 0.0, 1.0))
    if label == "Healthy Arc":
        value = 0.55 + 0.35 * rule_strength + 0.10 * (1.0 - anomaly_severity)
    else:
        corroboration = min(0.10, 0.05 * max(0, strong_rule_count - 1))
        value = 0.50 + 0.35 * rule_strength + 0.15 * anomaly_severity + corroboration
    return round(float(np.clip(value, 0.0, 1.0)), 5)


class InferencePipeline:
    def __init__(self, model_dir: str | os.PathLike = "models"):
        self.model_dir = Path(model_dir)
        self._profile_cache: dict[str, dict[str, Any]] = {}
        self._default_bundle = self._load_bundle(self.model_dir)
        self.scaler = self._default_bundle["scaler"]
        self.isolation_forest = self._default_bundle["isolation_forest"]
        self.thresholds = self._default_bundle["thresholds"]
        self.vae = self._default_bundle["vae"]
        self.detector = self._default_bundle["detector"]
        self.scaler_loaded = self.scaler is not None
        self.vae_loaded = self.vae is not None
        self.threshold_loaded = bool(self.thresholds)
        self.isolation_forest_loaded = self.isolation_forest is not None
        self.ready = all((
            self.scaler_loaded, self.vae_loaded, self.threshold_loaded,
            self.isolation_forest_loaded,
        ))

    @staticmethod
    def _profile_key(material: str, thickness_mm: float) -> str:
        clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(material)).strip("_") or "material"
        return f"{clean}_{float(thickness_mm):.2f}mm".replace(".", "p")

    def _load_bundle(self, path: Path) -> dict[str, Any]:
        threshold_path = path / "anomaly_threshold.json"
        thresholds = json.loads(threshold_path.read_text()) if threshold_path.exists() else {}
        expected = len(WindowFeatures.names())
        if thresholds and int(thresholds.get("feature_count", expected)) != expected:
            thresholds = {}

        scaler = None
        scaler_path = path / "scaler.pkl"
        if scaler_path.exists():
            candidate = joblib.load(scaler_path)
            if int(getattr(candidate, "n_features_in_", expected)) == expected:
                scaler = candidate

        isolation_forest = None
        forest_path = path / "isolation_forest.pkl"
        if forest_path.exists():
            candidate = joblib.load(forest_path)
            if int(getattr(candidate, "n_features_in_", expected)) == expected:
                isolation_forest = candidate

        vae = None
        vae_path = path / "vae.pt"
        if vae_path.exists() and thresholds:
            from .vae import load_vae
            candidate = load_vae(str(vae_path))
            if int(candidate.model.encoder[0].in_features) == expected:
                vae = candidate

        return {
            "thresholds": thresholds,
            "scaler": scaler,
            "isolation_forest": isolation_forest,
            "vae": vae,
            "detector": AnomalyDetector(thresholds, isolation_forest, vae),
        }

    def _bundle_for(self, material: str, thickness_mm: float) -> dict[str, Any]:
        key = self._profile_key(material, thickness_mm)
        if key not in self._profile_cache:
            profile_path = self.model_dir / "profiles" / key
            self._profile_cache[key] = self._load_bundle(profile_path) if profile_path.exists() else self._default_bundle
        return self._profile_cache[key]

    def health(self) -> dict[str, bool]:
        return {
            "model_ready": self.ready, "vae_loaded": self.vae_loaded,
            "scaler_loaded": self.scaler_loaded,
            "threshold_loaded": self.threshold_loaded,
            "isolation_forest_loaded": self.isolation_forest_loaded,
        }

    def _contributors(self, scaled: np.ndarray, vae: Any) -> list[dict[str, float | str]]:
        if vae is not None:
            importance = vae.reconstruction_contributions(scaled)
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

    def predict_features(self, features: WindowFeatures, material: str = "mild_steel", thickness_mm: float = 6.0, **__: Any) -> dict[str, Any]:
        bundle = self._bundle_for(material, thickness_mm)
        thresholds = bundle["thresholds"]
        reference = thresholds.get("feature_reference", {})
        physics = assess(features, reference)
        raw = features.to_vector().reshape(1, -1)
        scaler = bundle["scaler"]
        scaled = scaler.transform(raw) if scaler is not None else raw
        anomaly = bundle["detector"].score(scaled[0], physics.score)
        variance_q99 = float(reference.get("variance_v", {}).get("q99", max(features.variance_v, 1.0)))
        variance_health = 1.0 - min(1.0, features.variance_v / max(variance_q99, 1e-9))
        quality = compute_quality_index(
            physics.stability_score, variance_health, anomaly.vae_score,
            anomaly.isolation_score, physics.score,
        )
        status = physics.label
        contributors = self._contributors(scaled, bundle["vae"])
        severity = (
            "CRITICAL" if anomaly.level == "Critical" or quality["value"] < 30 else
            "WARNING" if anomaly.level == "Warning" or quality["value"] < 55 else
            "WATCH" if anomaly.level == "Watch" or quality["value"] < 75 else "NORMAL"
        )
        confidence = _confidence(
            physics.label, physics.rule_strength, physics.strong_rule_count,
            anomaly.score, anomaly.threshold,
        )
        return {
            # New healthy-only contract.
            "quality_score": quality["value"], "anomaly_score": anomaly.score,
            "status": status, "diagnosis": physics.diagnosis,
            "top_contributors": physics.top_contributors,
            # Existing frontend-compatible fields.
            "quality_index": quality["value"], "quality_category": quality["band"],
            "severity": severity, "anomaly_threshold": anomaly.threshold,
            "anomaly_detected": anomaly.is_anomaly,
            "anomaly_level": anomaly.level,
            "anomaly_stages": {
                "ewma": anomaly.ewma_score,
                "isolation_forest": anomaly.isolation_score,
                "physics_vae": anomaly.vae_score,
            },
            "physics_label": physics.label, "ml_label": physics.label,
            "prediction": physics.label,
            "confidence": confidence,
            **physics.risk_scores,
            "quality_breakdown": physics.quality_breakdown,
            "top_features": contributors, "explanation": physics.diagnosis,
            "recommendation": physics.recommendation,
            "stability_score": round(physics.stability_score, 4),
            "voltage_features": features.to_dict(), "model_ready": self.ready,
        }

    def predict(self, voltage: list[float], *args: Any, **kwargs: Any) -> dict:
        return self.predict_features(extract(voltage), *args, **kwargs)
