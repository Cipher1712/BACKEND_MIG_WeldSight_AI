"""Material- and thickness-aware physics classifier."""
from __future__ import annotations
from .features import WindowFeatures


def classify(features: WindowFeatures, score: float, threshold: float, material: str, thickness_mm: float) -> dict:
    ratio = score / max(threshold, 1e-6)

    heat_factor = {"mild_steel": 1.0, "stainless": 0.9, "aluminium": 1.2, "copper_alloy": 1.3,
                   "hsla": 1.05, "cast_iron": 0.95, "custom": 1.0}.get(material, 1.0)
    thick_factor = max(0.7, min(1.5, thickness_mm / 6.0))
    expected_mean = 22.0 * heat_factor * (0.9 + 0.1 * thick_factor)

    severity = "NORMAL"; label = "stable_arc"; display = "Stable Arc"
    text = "Within learned operating envelope for this material/thickness."
    causes: list[str] = []; actions: list[str] = []

    if ratio >= 1.8 or features.crest_factor > 1.35:
        severity, label, display = "CRITICAL", "short_circuit_abnormality", "Short-Circuit Abnormality"
        text = f"Crest factor {features.crest_factor:.2f} and {features.sc_count} short-circuit events exceed safe envelope."
        causes = ["Excessive contact-tip-to-work distance", "Wire feeder slippage", "Insufficient shielding gas"]
        actions = ["Reduce CTWD to 10-15 mm", "Inspect wire liner and drive rolls", "Verify gas flow >= 12 L/min"]
    elif features.mean_v > expected_mean * 1.18:
        severity, label, display = "WARNING", "heat_input_high", "Heat Input High"
        text = f"Mean voltage {features.mean_v:.1f} V above expected for {material} {thickness_mm} mm."
        causes = ["Voltage setpoint too high", "Travel speed too slow"]
        actions = ["Reduce voltage by 1-2 V", "Increase travel speed"]
    elif features.mean_v < expected_mean * 0.82:
        severity, label, display = "WARNING", "heat_input_low", "Heat Input Low"
        text = f"Mean voltage {features.mean_v:.1f} V below expected - risk of cold lap."
        causes = ["Voltage setpoint too low", "Long stick-out"]
        actions = ["Increase voltage", "Reduce stick-out"]
    elif ratio >= 1.25 or features.std_v > 1.6:
        severity, label, display = "WARNING", "arc_instability", "Arc Instability"
        text = f"sigma V={features.std_v:.2f} high relative to mean."
        causes = ["Magnetic arc blow", "Contaminated surface", "Inconsistent torch angle"]
        actions = ["Reposition ground clamp", "Clean workpiece", "Stabilise torch angle at 10-15 deg drag"]
    elif ratio >= 0.95:
        severity, label, display = "INFO", "transfer_change", "Transfer Mode Change"
        text = "Score approaching threshold - transfer regime may be shifting."
        causes = ["Voltage drift", "Thickness variation"]
        actions = ["Confirm voltage setpoint", "Check WFS calibration"]

    return {"severity": severity, "physics_label": label, "display_label": display,
            "physics_text": text, "possible_causes": causes, "recommended_actions": actions,
            "features": features.to_dict(), "material": material, "thickness_mm": thickness_mm}