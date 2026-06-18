"""Voltage-only features for 64-sample, 750 Hz welding windows."""
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
    rms_v: float
    variance_v: float
    std_v: float
    min_v: float
    max_v: float
    peak_to_peak_v: float
    energy: float
    crest_factor: float
    skewness: float
    kurtosis: float
    drift_v: float
    slope_v_per_s: float
    arc_stability_index: float
    ripple: float
    consistency: float
    noise_index: float
    short_circuit_count: int
    short_circuit_density: float
    avg_short_duration_ms: float
    short_circuit_ratio: float
    dominant_frequency_hz: float
    spectral_entropy: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    low_frequency_ratio: float
    high_frequency_ratio: float
    voltage_drop_density: float
    spike_density: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        return np.asarray([getattr(self, f.name) for f in fields(self)], dtype=np.float32)

    @property
    def sc_count(self) -> int:
        return self.short_circuit_count

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.empty(0, dtype=int)
    edges = np.flatnonzero(np.diff(np.pad(mask.astype(np.int8), (1, 1))))
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
    mean, median = float(v.mean()), float(np.median(v))
    centered = v - mean
    variance = float(np.mean(centered * centered))
    std = math.sqrt(max(variance, 0.0))
    rms = float(np.sqrt(np.mean(v * v)))
    minimum, maximum = float(v.min()), float(v.max())
    peak_to_peak = maximum - minimum
    energy = float(np.mean(v * v))
    crest = float(np.max(np.abs(v)) / (rms + EPS))
    skewness = float(np.mean(centered**3) / (std**3 + EPS)) if n > 2 else 0.0
    kurtosis = float(np.mean(centered**4) / (variance**2 + EPS) - 3.0) if n > 3 else 0.0
    quarter = max(1, n // 4)
    drift = float(np.mean(v[-quarter:]) - np.mean(v[:quarter]))
    x = np.arange(n, dtype=np.float64) / sampling_rate_hz
    slope = float(np.polyfit(x, v, 1)[0]) if n > 1 else 0.0

    dv = np.diff(v)
    ripple = float(np.sqrt(np.mean(dv * dv))) if dv.size else 0.0
    noise = float(np.median(np.abs(dv - np.median(dv))) * 1.4826) if dv.size else 0.0
    coefficient_of_variation = std / (abs(mean) + EPS)
    consistency = float(100.0 / (1.0 + 4.0 * coefficient_of_variation + ripple / (abs(mean) + EPS)))

    # Short-circuit threshold is relative to each operating voltage. The 8 V
    # cap is supported by the real datasets, whose stable arc centers are 15-20 V.
    threshold = float(sc_threshold) if sc_threshold is not None else min(8.0, max(2.0, median * 0.45))
    short_mask = v < threshold
    short_runs = _run_lengths(short_mask)
    short_count = int(short_runs.size)
    short_ratio = float(short_mask.mean())
    short_density = float(short_count * sampling_rate_hz / n)
    short_duration = float(short_runs.mean() * 1000.0 / sampling_rate_hz) if short_runs.size else 0.0
    robust_dv = noise + EPS
    drop_density = float(np.mean(dv < -max(1.0, 3.0 * robust_dv))) if dv.size else 0.0
    spike_density = float(np.mean(np.abs(dv) > max(1.5, 4.0 * robust_dv))) if dv.size else 0.0
    stability = float(100.0 * np.exp(-(
        2.5 * coefficient_of_variation + 1.6 * ripple / (abs(mean) + EPS) +
        2.0 * short_ratio + 1.2 * spike_density
    )))

    tapered = centered * np.hanning(n)
    magnitude = np.abs(np.fft.rfft(tapered))
    frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
    magnitude[0] = 0.0
    power = magnitude * magnitude
    total_power = float(power.sum()) + EPS
    probability = power / total_power
    nonzero = probability > 0
    entropy = float(-np.sum(probability[nonzero] * np.log2(probability[nonzero])) /
                    max(math.log2(probability.size), EPS))
    dominant = float(frequencies[int(np.argmax(power))])
    centroid = float(np.sum(frequencies * power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * power) / total_power))
    low_ratio = float(power[frequencies <= 50.0].sum() / total_power)
    high_ratio = float(power[frequencies >= 150.0].sum() / total_power)

    return WindowFeatures(
        mean, median, rms, variance, std, minimum, maximum, peak_to_peak,
        energy, crest, skewness, kurtosis, drift, slope, stability, ripple,
        consistency, noise, short_count, short_density, short_duration,
        short_ratio, dominant, entropy, centroid, bandwidth, low_ratio,
        high_ratio, drop_density, spike_density,
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
        position = float(distance[midpoint]) if distance is not None and midpoint < len(distance) else float(midpoint)
        yield position, extract(voltages[start:start + size], sampling_rate_hz=sampling_rate_hz)


def feature_matrix(rows: Sequence[WindowFeatures]) -> np.ndarray:
    return np.vstack([row.to_vector() for row in rows]) if rows else np.empty((0, len(WindowFeatures.names())))
