"""Baseline profile training pipeline."""
from __future__ import annotations
from typing import List, Dict
import numpy as np
from .features import WindowFeatures, windowize


def train_baseline(good_welds: List[Dict], material: str, thickness_mm: float) -> dict:
    all_features: List[WindowFeatures] = []
    all_voltages: List[float] = []
    scores: List[float] = []

    for w in good_welds:
        v = list(w.get("voltage", []))
        d = list(w.get("distance", [])) or list(range(len(v)))
        all_voltages.extend(v)
        for _, f in windowize(v, d, size=64, step=32):
            all_features.append(f)
            scores.append(f.std_v + max(0.0, f.crest_factor - 1.1) * 4.0)

    if not scores:
        raise ValueError("No usable windows in training set")

    arr = np.asarray(scores)
    mean_score = float(arr.mean())
    std_score = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    learned_k = float(max(2.0, min(4.5, 2.4 + std_score * 0.6)))
    rms_values = [f.rms_v for f in all_features]
    return {
        "material": material, "thickness_mm": thickness_mm,
        "learned_k": round(learned_k, 2), "mean_score": round(mean_score, 3),
        "std_score": round(std_score, 3),
        "voltage_min": round(float(min(all_voltages)), 2) if all_voltages else 0.0,
        "voltage_max": round(float(max(all_voltages)), 2) if all_voltages else 0.0,
        "rms_min": round(float(min(rms_values)), 2) if rms_values else 0.0,
        "rms_max": round(float(max(rms_values)), 2) if rms_values else 0.0,
        "trained_windows": len(scores),
    }