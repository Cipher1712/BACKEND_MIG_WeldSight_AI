"""Three-stage anomaly score fusion."""
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
    stage1_score: float
    stage2_score: float
    stage3_score: float
    threshold: float
    reconstruction_error: float = 0.0
    latent_distance: float = 0.0


class AnomalyDetector:
    def __init__(self, thresholds: dict[str, Any] | None = None, isolation_forest: Any = None, vae: Any = None):
        cfg = thresholds or {}
        self.cfg = cfg
        self.ewma = DynamicThreshold(float(cfg.get("ewma_k", 3.0)), floor=float(cfg.get("ewma_floor", 0.1)))
        self.isolation_forest = isolation_forest
        self.vae = vae

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))

    def score(self, scaled_features: np.ndarray, physics_score: float) -> AnomalyResult:
        x = np.asarray(scaled_features, dtype=np.float32).reshape(1, -1)
        stage1_raw = float(np.linalg.norm(x) / math.sqrt(x.shape[1]))
        ewma = self.ewma.update(stage1_raw)
        stage1 = self._sigmoid((stage1_raw - ewma["threshold"]) / max(ewma["sigma"], 0.25))

        stage2 = 0.0
        if self.isolation_forest is not None:
            stage2 = self._sigmoid(-float(self.isolation_forest.decision_function(x)[0]) * 8.0)

        stage3 = reconstruction = latent = 0.0
        if self.vae is not None:
            reconstruction, latent = self.vae.anomaly_components(x)
            rec_t = float(self.cfg.get("reconstruction_threshold", 1.0))
            lat_t = float(self.cfg.get("latent_threshold", 3.0))
            stage3 = 0.65 * self._sigmoid((reconstruction - rec_t) / max(rec_t * 0.2, 0.05))
            stage3 += 0.35 * self._sigmoid((latent - lat_t) / max(lat_t * 0.2, 0.1))

        fused = 0.25 * stage1
        if self.isolation_forest is not None:
            fused += 0.30 * stage2
        if self.vae is not None:
            fused += 0.35 * stage3
        fused += 0.10 * float(physics_score)
        threshold = float(self.cfg.get("anomaly_threshold", 0.60))
        return AnomalyResult(round(fused, 6), fused >= threshold, round(stage1, 6),
                             round(stage2, 6), round(stage3, 6), threshold,
                             round(reconstruction, 6), round(latent, 6))
