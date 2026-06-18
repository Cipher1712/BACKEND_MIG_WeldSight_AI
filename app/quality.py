"""Voltage-only weld quality score."""
from __future__ import annotations


def compute_quality_index(
    stability_score: float,
    variance_health: float,
    vae_score: float,
    isolation_score: float,
    physics_score: float,
) -> dict[str, int | str]:
    value = round(100.0 * (
        0.28 * max(0.0, min(1.0, stability_score / 100.0)) +
        0.15 * max(0.0, min(1.0, variance_health)) +
        0.32 * (1.0 - max(0.0, min(1.0, vae_score))) +
        0.15 * (1.0 - max(0.0, min(1.0, isolation_score))) +
        0.10 * (1.0 - max(0.0, min(1.0, physics_score)))
    ))
    category = (
        "Excellent" if value >= 90 else "Good" if value >= 75 else
        "Warning" if value >= 55 else "Poor" if value >= 30 else "Critical"
    )
    return {"value": int(value), "band": category}
