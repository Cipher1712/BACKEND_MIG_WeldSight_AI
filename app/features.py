"""Fast, deterministic voltage-window feature extraction."""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, fields
import math

import numpy as np

EPS = 1e-9
SAMPLING_RATE_HZ = 750.0
WINDOW_SIZE = 64
WINDOW_STRIDE = 32


@dataclass(slots=True)
class WindowFeatures:
    mean_v: float
    median_v: float
    min_v: float
    max_v: float
    rms_v: float
    variance_v: float
    std_v: float
    peak_to_peak_v: float
    skewness: float
    kurtosis: float
    energy: float
    crest_factor: float
    drift_v: float
    slope_v_per_s: float
    arc_stability_index: float
    short_circuit_count: int
    short_circuit_ratio: float
    avg_short_duration_ms: float
    voltage_drop_density: float
    spike_density: float
    arc_extinction_count: int
    instability_score: float
    dominant_frequency_hz: float
    fft_peak_magnitude: float
    fft_mean_magnitude: float
    spectral_entropy: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    low_frequency_ratio: float
    high_frequency_ratio: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        return np.asarray([getattr(self, f.name) for f in fields(self)], dtype=np.float32)

    @property
    def sc_count(self) -> int:
        """Compatibility alias for the original API."""
        return self.short_circuit_count

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.empty(0, dtype=int)
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return edges[1::2] - edges[::2]


def extract(
    voltages: Iterable[float],
    sc_threshold: float | None = None,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
) -> WindowFeatures:
    v = np.asarray(list(voltages), dtype=np.float64)
    if v.size == 0:
        return WindowFeatures(*([0.0] * len(WindowFeatures.names())))
    if not np.isfinite(v).all():
        finite = v[np.isfinite(v)]
        fill = float(np.median(finite)) if finite.size else 0.0
        v = np.nan_to_num(v, nan=fill, posinf=fill, neginf=fill)

    n = v.size
    mean = float(v.mean())
    median = float(np.median(v))
    centered = v - mean
    variance = float(np.mean(centered * centered))
    std = math.sqrt(max(variance, 0.0))
    rms = float(np.sqrt(np.mean(v * v)))
    v_min, v_max = float(v.min()), float(v.max())
    ptp = v_max - v_min
    skew = float(np.mean(centered**3) / (std**3 + EPS)) if n > 2 else 0.0
    kurt = float(np.mean(centered**4) / (variance**2 + EPS) - 3.0) if n > 3 else 0.0
    energy = float(np.mean(v * v))
    crest = float(np.max(np.abs(v)) / (rms + EPS))
    drift = float(np.mean(v[-max(1, n // 4):]) - np.mean(v[:max(1, n // 4)]))
    x = np.arange(n, dtype=np.float64) / sampling_rate_hz
    slope = float(np.polyfit(x, v, 1)[0]) if n > 1 else 0.0

    # A dynamic threshold follows the local arc level while retaining the legacy
    # 8 V floor used by existing MMAW/MIG installations.
    threshold = float(sc_threshold) if sc_threshold is not None else max(8.0, mean * 0.45)
    short_mask = v < threshold
    runs = _run_lengths(short_mask)
    sc_count = int(runs.size)
    sc_ratio = float(short_mask.mean())
    avg_short_ms = float(runs.mean() * 1000.0 / sampling_rate_hz) if runs.size else 0.0
    dv = np.diff(v)
    robust_scale = float(np.median(np.abs(dv - np.median(dv))) * 1.4826 + EPS) if dv.size else EPS
    drop_density = float(np.mean(dv < -max(1.0, 3.0 * robust_scale))) if dv.size else 0.0
    spike_density = float(np.mean(np.abs(dv) > max(1.5, 4.0 * robust_scale))) if dv.size else 0.0
    extinction_threshold = max(2.0, mean * 0.12)
    extinction_mask = v < extinction_threshold
    extinction_count = int(_run_lengths(extinction_mask).size)
    cv = std / (abs(mean) + EPS)
    stability = float(100.0 * np.exp(-(2.8 * cv + 2.2 * sc_ratio + 1.8 * spike_density)))
    instability = float(np.clip(100.0 - stability + 20.0 * extinction_count, 0.0, 100.0))

    if n > 1:
        tapered = centered * np.hanning(n)
        magnitudes = np.abs(np.fft.rfft(tapered))
        frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
        magnitudes[0] = 0.0
        power = magnitudes * magnitudes
        total_power = float(power.sum()) + EPS
        probability = power / total_power
        nonzero = probability > 0
        entropy = float(-np.sum(probability[nonzero] * np.log2(probability[nonzero])) /
                        max(math.log2(probability.size), EPS))
        dominant_idx = int(np.argmax(power))
        dominant = float(frequencies[dominant_idx])
        centroid = float(np.sum(frequencies * power) / total_power)
        bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * power) / total_power))
        low_ratio = float(power[frequencies <= 50.0].sum() / total_power)
        high_ratio = float(power[frequencies >= 150.0].sum() / total_power)
        fft_peak = float(magnitudes.max() / n)
        fft_mean = float(magnitudes.mean() / n)
    else:
        dominant = fft_peak = fft_mean = entropy = centroid = bandwidth = low_ratio = high_ratio = 0.0

    return WindowFeatures(
        mean, median, v_min, v_max, rms, variance, std, ptp, skew, kurt,
        energy, crest, drift, slope, stability, sc_count, sc_ratio,
        avg_short_ms, drop_density, spike_density, extinction_count,
        instability, dominant, fft_peak, fft_mean, entropy, centroid,
        bandwidth, low_ratio, high_ratio,
    )


def windowize(
    voltages: Sequence[float],
    distance: Sequence[float] | None = None,
    size: int = WINDOW_SIZE,
    step: int = WINDOW_STRIDE,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
) -> Iterator[tuple[float, WindowFeatures]]:
    if size < 16 or step < 1 or step > size:
        raise ValueError("window size must be >=16 and stride must be in [1, size]")
    for start in range(0, len(voltages) - size + 1, step):
        midpoint = start + size // 2
        d_mid = (
            float(distance[midpoint])
            if distance is not None and midpoint < len(distance)
            else float(midpoint)
        )
        yield d_mid, extract(voltages[start:start + size], sampling_rate_hz=sampling_rate_hz)


def feature_matrix(rows: Sequence[WindowFeatures]) -> np.ndarray:
    return np.vstack([row.to_vector() for row in rows]) if rows else np.empty((0, len(WindowFeatures.names())))
