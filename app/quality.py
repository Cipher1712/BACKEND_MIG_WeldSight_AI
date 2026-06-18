"""Calibrated, monotonic weld quality index."""
from __future__ import annotations


def compute_quality_index(
    physics_score: float,
    anomaly_score: float,
    classifier_confidence: float,
    stability_score: float,
    predicted_stable: bool = False,
) -> dict[str, int | str]:
    physics_health = 1.0 - max(0.0, min(1.0, physics_score))
    anomaly_health = 1.0 - max(0.0, min(1.0, anomaly_score))
    stability_health = max(0.0, min(1.0, stability_score / 100.0))
    confidence = max(0.0, min(1.0, classifier_confidence))
    classification_health = confidence if predicted_stable else 1.0 - confidence
    value = round(100.0 * (
        0.30 * physics_health + 0.35 * anomaly_health +
        0.25 * stability_health + 0.10 * classification_health
    ))
    category = (
        "Excellent" if value >= 90 else "Good" if value >= 75 else
        "Warning" if value >= 55 else "Poor" if value >= 30 else "Critical"
    )
    return {"value": int(value), "band": category}
