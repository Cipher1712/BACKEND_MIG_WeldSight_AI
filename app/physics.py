"""Physics-based diagnosis using healthy-data reference ranges."""
from __future__ import annotations

from dataclasses import dataclass

from .features import WindowFeatures


@dataclass(slots=True)
class PhysicsAssessment:
    label: str
    diagnosis: str
    score: float
    stability_score: float
    recommendation: str


def _bound(reference: dict, feature: str, quantile: str, default: float) -> float:
    return float(reference.get(feature, {}).get(quantile, default))


def assess(features: WindowFeatures, reference: dict | None = None) -> PhysicsAssessment:
    ref = reference or {}
    high_variance = features.variance_v > _bound(ref, "variance_v", "q99", 16.0)
    high_ripple = features.ripple > _bound(ref, "ripple", "q99", 8.0)
    high_energy = features.energy > _bound(ref, "energy", "q99", 900.0)
    high_voltage = features.mean_v > _bound(ref, "mean_v", "q99", 35.0)
    low_energy = features.energy < _bound(ref, "energy", "q01", 100.0)
    unstable = features.arc_stability_index < _bound(ref, "arc_stability_index", "q01", 35.0)
    high_short = features.short_circuit_density > _bound(ref, "short_circuit_density", "q99", 60.0)

    score = min(1.0, (
        max(0.0, 1.0 - features.arc_stability_index / 100.0) * 0.40 +
        min(1.0, features.short_circuit_ratio / 0.40) * 0.25 +
        min(1.0, features.ripple / max(abs(features.mean_v), 1.0)) * 0.20 +
        min(1.0, features.noise_index / max(abs(features.mean_v), 1.0)) * 0.15
    ))
    if high_energy and high_voltage:
        return PhysicsAssessment(
            "burn_through_risk",
            "High voltage and high electrical energy exceed the healthy operating envelope.",
            score, features.arc_stability_index,
            "Reduce voltage or increase travel speed and verify the welding procedure.",
        )
    if low_energy and unstable:
        return PhysicsAssessment(
            "cold_arc_risk",
            "Low electrical energy combined with unstable voltage indicates a cold-arc risk.",
            score, features.arc_stability_index,
            "Check voltage setpoint, stick-out, grounding, and travel speed.",
        )
    if high_short:
        return PhysicsAssessment(
            "transfer_irregularity",
            "Short-circuit event density is above the healthy transfer envelope.",
            score, features.arc_stability_index,
            "Inspect wire feed, contact-tip distance, polarity, and shielding gas.",
        )
    if high_variance and high_ripple:
        return PhysicsAssessment(
            "arc_instability",
            "High voltage variance and ripple indicate arc instability.",
            score, features.arc_stability_index,
            "Stabilize torch angle and arc length; clean the joint and verify grounding.",
        )
    return PhysicsAssessment(
        "healthy_arc", "Voltage behavior is within the learned healthy operating envelope.",
        score, features.arc_stability_index, "Maintain the current welding parameters.",
    )
