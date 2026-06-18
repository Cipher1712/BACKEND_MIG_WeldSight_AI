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

        # VAE is primary. Physics and EWMA provide responsiveness and the
        # Isolation Forest provides a geometrically different corroboration.
        score = (
            0.55 * vae_score + 0.20 * isolation_score +
            0.15 * ewma_score + 0.10 * float(np.clip(physics_score, 0.0, 1.0))
        )
        threshold = float(self.cfg.get("anomaly_threshold", 0.65))
        return AnomalyResult(
            round(score, 6), score >= threshold, round(ewma_score, 6),
            round(isolation_score, 6), round(vae_score, 6), threshold,
            round(reconstruction_error, 6), round(latent_distance, 6),
        )
