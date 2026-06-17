"""Window-level feature extraction from raw voltage samples."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List
import math
import numpy as np


@dataclass
class WindowFeatures:
    mean_v: float
    std_v: float
    min_v: float
    max_v: float
    rms_v: float
    sc_count: int
    crest_factor: float

    def to_dict(self) -> dict:
        return asdict(self)


def extract(voltages: Iterable[float], sc_threshold: float = 8.0) -> WindowFeatures:
    v = np.asarray(list(voltages), dtype=float)
    if v.size == 0:
        return WindowFeatures(0, 0, 0, 0, 0, 0, 0)
    mean_v = float(v.mean())
    std_v = float(v.std(ddof=1)) if v.size > 1 else 0.0
    rms_v = float(math.sqrt(np.mean(v * v)))
    sc = int(np.sum(v < sc_threshold))
    crest = float(np.max(np.abs(v)) / max(rms_v, 1e-6))
    return WindowFeatures(mean_v, std_v, float(v.min()), float(v.max()), rms_v, sc, crest)


def windowize(voltages: List[float], distance: List[float], size: int = 64, step: int = 32):
    n = len(voltages)
    for i in range(0, n - size + 1, step):
        w = voltages[i : i + size]
        d_mid = distance[i + size // 2] if i + size // 2 < len(distance) else (i + size // 2)
        yield d_mid, extract(w)