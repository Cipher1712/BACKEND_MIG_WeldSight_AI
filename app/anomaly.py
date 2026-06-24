"""EWMA, Isolation Forest, and primary VAE anomaly scoring."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .dynamic_threshold import DynamicThreshold


@dataclass(slots=True)
class AnomalyResult:
    score: float
    is_anomaly: bool
    level: str
    ewma_score: float
    isolation_score: float
    vae_score: float
    threshold: float
    reconstruction_error: float
    latent_distance: float


def _calibrate(value: float, reference: dict[str, float]) -> float:
    median = float(reference.get("median", 0.0))
    high = float(reference.get("q995", reference.get("q95", median + 1.0)))
    return float(np.clip((value - median) / max(high - median, 1e-9), 0.0, 1.0))


def _level(score: float, bands: dict[str, float]) -> str:
    watch = float(bands.get("watch", 0.60))
    warning = float(bands.get("warning", 0.78))
    normal = float(bands.get("normal", min(watch, 0.45)))
    if score >= warning:
        return "Critical"
    if score >= watch:
        return "Warning"
    if score >= normal:
        return "Watch"
    return "Normal"


class AnomalyDetector:
    def __init__(self, thresholds: dict[str, Any] | None = None, isolation_forest: Any = None, vae: Any = None):
        self.cfg = thresholds or {}
        self.isolation_forest = isolation_forest
        self.vae = vae
        self.ewma = DynamicThreshold(float(self.cfg.get("ewma_k", 3.0)), floor=0.05)

    def score(self, scaled_features: np.ndarray, physics_score: float) -> AnomalyResult:
        x = np.asarray(scaled_features, dtype=np.float32).reshape(1, -1)
        magnitude = float(np.linalg.norm(x) / math.sqrt(x.shape[1]))
        adaptive = self.ewma.update(magnitude)
        ewma_score = float(np.clip(
            (magnitude - adaptive["ewma"]) / max(adaptive["threshold"] - adaptive["ewma"], 0.25),
            0.0, 1.0,
        ))

        isolation_score = 0.0
        if self.isolation_forest is not None:
            isolation_raw = -float(self.isolation_forest.decision_function(x)[0])
            isolation_score = _calibrate(isolation_raw, self.cfg.get("isolation_forest", {}))

        reconstruction_error = latent_distance = vae_score = 0.0
        if self.vae is not None:
            reconstruction_error, latent_distance = self.vae.anomaly_components(x)
            reconstruction_score = _calibrate(
                reconstruction_error, self.cfg.get("reconstruction", {})
            )
            latent_score = _calibrate(
                latent_distance, self.cfg.get("latent_distance", {})
            )
            vae_score = 0.75 * reconstruction_score + 0.25 * latent_score

        weights = self.cfg.get("fusion_weights", {})
        vae_w = float(weights.get("vae", 0.42))
        forest_w = float(weights.get("isolation_forest", 0.24))
        ewma_w = float(weights.get("ewma", 0.18))
        physics_w = float(weights.get("physics", 0.16))
        total_w = max(vae_w + forest_w + ewma_w + physics_w, 1e-9)
        score = (
            vae_w * vae_score + forest_w * isolation_score +
            ewma_w * ewma_score + physics_w * float(np.clip(physics_score, 0.0, 1.0))
        ) / total_w
        threshold = float(self.cfg.get("anomaly_threshold", 0.65))
        level = _level(score, self.cfg.get("score_bands", {}))
        return AnomalyResult(
            round(score, 6), score >= threshold, level, round(ewma_score, 6),
            round(isolation_score, 6), round(vae_score, 6), threshold,
            round(reconstruction_error, 6), round(latent_distance, 6),
        )
