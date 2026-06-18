"""Voltage-only welding physics constraints and interpretable labels."""
from __future__ import annotations

from dataclasses import dataclass

from .features import WindowFeatures


@dataclass(slots=True)
class PhysicsAssessment:
    label: str
    score: float
    stability_score: float
    violations: dict[str, float]
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "physics_label": self.label,
            "physics_score": self.score,
            "stability_score": self.stability_score,
            "physics_violations": self.violations,
            "recommendation": self.recommendation,
        }


def assess(features: WindowFeatures, material: str = "mild_steel", thickness_mm: float = 6.0) -> PhysicsAssessment:
    heat_factor = {
        "mild_steel": 1.0, "stainless": 0.92, "aluminium": 1.18,
        "copper_alloy": 1.25, "hsla": 1.04, "cast_iron": 0.95,
    }.get(material, 1.0)
    thickness_factor = min(1.25, max(0.8, thickness_mm / 6.0))
    expected_v = 22.0 * heat_factor * (0.92 + 0.08 * thickness_factor)

    violations = {
        "arc_stability": max(0.0, (70.0 - features.arc_stability_index) / 70.0),
        "variance": max(0.0, (features.std_v - 1.5) / 3.0),
        "short_circuit": max(0.0, (features.short_circuit_ratio - 0.18) / 0.45),
        "spectral_entropy": max(0.0, (features.spectral_entropy - 0.72) / 0.28),
    }
    score = min(1.0, 0.35 * violations["arc_stability"] + 0.25 * violations["variance"] +
                0.25 * violations["short_circuit"] + 0.15 * violations["spectral_entropy"])
    label = "stable_arc"
    recommendation = "Maintain current welding parameters."
    if features.arc_extinction_count:
        label, recommendation = "abnormal_arc_behaviour", "Check grounding, electrode continuity, and arc length."
    elif features.short_circuit_ratio > 0.32 or features.short_circuit_count > 8:
        label, recommendation = "short_circuit_instability", "Inspect wire feed and reduce excessive contact-tip distance."
    elif features.mean_v > expected_v * 1.18:
        label, recommendation = "heat_input_high", "Reduce voltage or increase travel speed; verify against the WPS."
    elif features.mean_v < expected_v * 0.82:
        label, recommendation = "heat_input_low", "Increase voltage or reduce travel speed; verify against the WPS."
    elif features.spike_density > 0.18:
        label, recommendation = "excessive_spatter", "Check polarity, shielding gas, wire feed, and voltage balance."
    elif score > 0.30:
        label, recommendation = "arc_instability", "Clean the workpiece and stabilize torch angle and arc length."
    return PhysicsAssessment(label, round(score, 5), round(features.arc_stability_index, 3),
                             {k: round(min(1.0, v), 5) for k, v in violations.items()}, recommendation)
