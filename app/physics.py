"""Physics-based diagnosis using healthy-data reference ranges."""
from __future__ import annotations

from dataclasses import dataclass

from .features import WindowFeatures


LABELS = {
    "healthy_arc": "Healthy Arc",
    "arc_instability": "Arc Instability",
    "transfer_irregularity": "Spatter Risk",
    "burn_through_risk": "Burn Through Risk",
    "cold_arc_risk": "Low Heat Input Risk",
}

DIAGNOSES = {
    "Healthy Arc": "Voltage behavior remains within the learned healthy operating envelope.",
    "Arc Instability": "Large voltage fluctuations and ripple indicate unstable arc behavior.",
    "Spatter Risk": "Elevated short-circuit activity suggests increased spatter probability.",
    "Burn Through Risk": "Sustained high energy and elevated voltage indicate excessive heat input risk.",
    "Low Heat Input Risk": "Insufficient energy and reduced arc stability suggest inadequate heat input.",
}

RECOMMENDATIONS = {
    "Healthy Arc": "Maintain the current welding parameters.",
    "Arc Instability": "Check grounding, arc length, and electrode condition.",
    "Spatter Risk": "Review transfer conditions and consumable settings.",
    "Burn Through Risk": "Reduce heat input or increase travel speed.",
    "Low Heat Input Risk": "Increase heat input or reduce travel speed.",
}

RISK_LABELS = {
    "arc_instability_score": "Arc Instability",
    "spatter_risk_score": "Spatter Risk",
    "burn_through_risk_score": "Burn Through Risk",
    "low_heat_input_score": "Low Heat Input Risk",
}


@dataclass(slots=True)
class PhysicsAssessment:
    label: str
    diagnosis: str
    score: float
    stability_score: float
    recommendation: str
    risk_scores: dict[str, float]
    top_contributors: list[str]
    rule_strength: float
    strong_rule_count: int
    quality_breakdown: dict[str, float]


def _bound(reference: dict, feature: str, quantile: str, default: float) -> float:
    return float(reference.get(feature, {}).get(quantile, default))


def _risk_high(reference: dict, feature: str, value: float, q50_default: float, q99_default: float) -> float:
    midpoint = _bound(reference, feature, "q50", q50_default)
    upper = _bound(reference, feature, "q99", q99_default)
    return round(max(0.0, min(1.0, (value - midpoint) / max(upper - midpoint, 1e-9))), 5)


def _risk_low(reference: dict, feature: str, value: float, q01_default: float, q50_default: float) -> float:
    lower = _bound(reference, feature, "q01", q01_default)
    midpoint = _bound(reference, feature, "q50", q50_default)
    return round(max(0.0, min(1.0, (midpoint - value) / max(midpoint - lower, 1e-9))), 5)


def _weighted(parts: tuple[tuple[float, float], ...]) -> float:
    return round(max(0.0, min(1.0, sum(weight * value for weight, value in parts))), 5)


def _risk_scores(features: WindowFeatures, reference: dict) -> dict[str, float]:
    high_variance = _risk_high(reference, "variance_v", features.variance_v, 4.0, 16.0)
    high_ripple = _risk_high(reference, "ripple", features.ripple, 2.0, 8.0)
    high_energy = _risk_high(reference, "energy", features.energy, 300.0, 900.0)
    high_voltage = _risk_high(reference, "mean_v", features.mean_v, 18.0, 35.0)
    low_energy = _risk_low(reference, "energy", features.energy, 100.0, 300.0)
    low_voltage = _risk_low(reference, "mean_v", features.mean_v, 12.0, 18.0)
    low_stability = _risk_low(reference, "arc_stability_index", features.arc_stability_index, 35.0, 80.0)
    high_short_density = _risk_high(
        reference, "short_circuit_density", features.short_circuit_density, 5.0, 60.0
    )
    high_short_ratio = _risk_high(reference, "short_circuit_ratio", features.short_circuit_ratio, 0.02, 0.40)

    return {
        "arc_instability_score": _weighted((
            (0.45, high_variance),
            (0.30, high_ripple),
            (0.25, low_stability),
        )),
        "spatter_risk_score": _weighted((
            (0.55, high_short_density),
            (0.30, high_short_ratio),
            (0.15, high_ripple),
        )),
        "burn_through_risk_score": _weighted((
            (0.50, high_energy),
            (0.30, high_voltage),
            (0.20, high_variance),
        )),
        "low_heat_input_score": _weighted((
            (0.45, low_energy),
            (0.35, low_voltage),
            (0.20, low_stability),
        )),
    }


def _top_contributors(features: WindowFeatures, reference: dict, label: str, risk_scores: dict[str, float]) -> list[str]:
    high_variance = _risk_high(reference, "variance_v", features.variance_v, 4.0, 16.0)
    high_ripple = _risk_high(reference, "ripple", features.ripple, 2.0, 8.0)
    high_energy = _risk_high(reference, "energy", features.energy, 300.0, 900.0)
    high_voltage = _risk_high(reference, "mean_v", features.mean_v, 18.0, 35.0)
    low_energy = _risk_low(reference, "energy", features.energy, 100.0, 300.0)
    low_voltage = _risk_low(reference, "mean_v", features.mean_v, 12.0, 18.0)
    low_stability = _risk_low(reference, "arc_stability_index", features.arc_stability_index, 35.0, 80.0)
    high_short_density = _risk_high(
        reference, "short_circuit_density", features.short_circuit_density, 5.0, 60.0
    )
    high_short_ratio = _risk_high(reference, "short_circuit_ratio", features.short_circuit_ratio, 0.02, 0.40)
    high_spike = _risk_high(reference, "spike_density", features.spike_density, 0.01, 0.25)

    if label == "Healthy Arc":
        healthy_terms = [
            (1.0 - risk_scores["arc_instability_score"], "Stable Arc Behavior"),
            (1.0 - risk_scores["spatter_risk_score"], "Low Short-Circuit Activity"),
            (1.0 - risk_scores["burn_through_risk_score"], "Controlled Heat Input"),
            (1.0 - risk_scores["low_heat_input_score"], "Adequate Heat Input"),
        ]
        return [name for _, name in sorted(healthy_terms, reverse=True)[:3]]

    terms = [
        (high_variance, "High Voltage Variance"),
        (high_ripple, "Elevated Ripple"),
        (low_stability, "Low Arc Stability"),
        (high_short_density, "High Short-Circuit Density"),
        (high_short_ratio, "High Short-Circuit Ratio"),
        (high_energy, "High Electrical Energy"),
        (high_voltage, "Elevated Mean Voltage"),
        (low_energy, "Low Electrical Energy"),
        (low_voltage, "Reduced Mean Voltage"),
        (high_spike, "Elevated Voltage Spikes"),
    ]
    selected = [name for value, name in sorted(terms, reverse=True) if value > 0.0]
    return (selected + ["Stable Arc Behavior", "Controlled Heat Input", "Low Short-Circuit Activity"])[:3]


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
        min(1.0, features.spike_density / 0.25) * 0.15
    ))
    quality_breakdown = {
        "stability": 0.40,
        "short_circuit": 0.25,
        "ripple": 0.20,
        "spike_density": 0.15,
    }
    risks = _risk_scores(features, ref)
    if high_energy and high_voltage:
        label = LABELS["burn_through_risk"]
        rule_strength = risks["burn_through_risk_score"]
        strong_rule_count = int(high_energy) + int(high_voltage) + int(high_variance)
        return PhysicsAssessment(label, DIAGNOSES[label], score, features.arc_stability_index,
                                 RECOMMENDATIONS[label], risks, _top_contributors(features, ref, label, risks),
                                 rule_strength, strong_rule_count, quality_breakdown)
    if low_energy and unstable:
        label = LABELS["cold_arc_risk"]
        rule_strength = risks["low_heat_input_score"]
        strong_rule_count = int(low_energy) + int(unstable)
        return PhysicsAssessment(label, DIAGNOSES[label], score, features.arc_stability_index,
                                 RECOMMENDATIONS[label], risks, _top_contributors(features, ref, label, risks),
                                 rule_strength, strong_rule_count, quality_breakdown)
    if high_short:
        label = LABELS["transfer_irregularity"]
        rule_strength = risks["spatter_risk_score"]
        strong_rule_count = int(high_short) + int(features.short_circuit_ratio > _bound(ref, "short_circuit_ratio", "q99", 0.40))
        return PhysicsAssessment(label, DIAGNOSES[label], score, features.arc_stability_index,
                                 RECOMMENDATIONS[label], risks, _top_contributors(features, ref, label, risks),
                                 rule_strength, strong_rule_count, quality_breakdown)
    if high_variance and high_ripple:
        label = LABELS["arc_instability"]
        rule_strength = risks["arc_instability_score"]
        strong_rule_count = int(high_variance) + int(high_ripple) + int(unstable)
        return PhysicsAssessment(label, DIAGNOSES[label], score, features.arc_stability_index,
                                 RECOMMENDATIONS[label], risks, _top_contributors(features, ref, label, risks),
                                 rule_strength, strong_rule_count, quality_breakdown)
    top_risk_key, top_risk_value = max(risks.items(), key=lambda item: item[1])
    if top_risk_value >= 0.65:
        label = RISK_LABELS[top_risk_key]
        strong_rule_count = sum(value >= 0.65 for value in risks.values())
        return PhysicsAssessment(label, DIAGNOSES[label], score, features.arc_stability_index,
                                 RECOMMENDATIONS[label], risks, _top_contributors(features, ref, label, risks),
                                 top_risk_value, strong_rule_count, quality_breakdown)
    label = LABELS["healthy_arc"]
    max_risk = max(risks.values()) if risks else 0.0
    return PhysicsAssessment(label, DIAGNOSES[label], score, features.arc_stability_index,
                             RECOMMENDATIONS[label], risks, _top_contributors(features, ref, label, risks),
                             round(1.0 - max_risk, 5), 0, quality_breakdown)
