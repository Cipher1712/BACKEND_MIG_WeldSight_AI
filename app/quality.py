def compute_quality_index(std_v: float, sc_count: int, crest_factor: float, score: float, ewma: float) -> dict:
    def clamp01(x): return max(0.0, min(1.0, x))
    electrical = clamp01(1 - min(std_v, 3.0) / 3.0) * 100
    transfer = clamp01(1 - min(abs(sc_count - 2), 6) / 6.0) * 100
    arc = clamp01(1 - min(abs(crest_factor - 1.1), 0.5) / 0.5) * 100
    drift = clamp01(1 - min(abs(score - ewma), 4.0) / 4.0) * 100
    value = round(0.4 * electrical + 0.3 * transfer + 0.2 * arc + 0.1 * drift)
    band = "Excellent" if value >= 85 else "Good" if value >= 70 else "Monitor" if value >= 50 else "Poor"
    return {"value": value, "band": band}