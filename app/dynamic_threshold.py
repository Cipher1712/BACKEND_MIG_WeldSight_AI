"""EWMA + learned_k dynamic threshold engine."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ThresholdState:
    mean: float = 0.0
    variance: float = 0.0
    n: int = 0


class DynamicThreshold:
    def __init__(self, learned_k: float, ewma_alpha: float = 0.15, var_alpha: float = 0.08, floor: float = 3.0):
        self.k = learned_k
        self.ewma_alpha = ewma_alpha
        self.var_alpha = var_alpha
        self.floor = floor
        self.state = ThresholdState()

    def update(self, score: float) -> dict:
        s = self.state
        s.n += 1
        if s.n == 1:
            s.mean = score
            s.variance = 0.0
        else:
            dev = score - s.mean
            s.mean = self.ewma_alpha * score + (1 - self.ewma_alpha) * s.mean
            s.variance = (1 - self.var_alpha) * s.variance + self.var_alpha * dev * dev
        sigma = math.sqrt(s.variance)
        adaptive = s.mean + self.k * sigma
        threshold = max(adaptive, self.floor * 0.85)
        return {"threshold": round(threshold, 3), "ewma": round(s.mean, 3), "sigma": round(sigma, 3), "k": self.k, "n": s.n}