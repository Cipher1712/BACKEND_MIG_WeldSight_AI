"""Per-sample timestamp and distance normalization for telemetry packets."""
from __future__ import annotations

from typing import Any

from .features import SAMPLING_RATE_HZ

SAMPLE_PERIOD_MS = 1000.0 / SAMPLING_RATE_HZ


def canonical_distance_source(value: str | None) -> str:
    clean = str(value or "").strip().lower()
    if clean == "encoder":
        return "Encoder"
    if clean == "relative":
        return "Relative"
    return "Estimated"


def api_distance_source(value: str | None) -> str:
    return canonical_distance_source(value).lower()


def packet_samples(packet: dict[str, Any]) -> list[dict[str, float | int | str | None]]:
    samples = [float(value) for value in packet["voltage"]]
    timestamps = packet.get("timestamps_ms")
    if isinstance(timestamps, list) and len(timestamps) == len(samples):
        sample_times = [int(value) for value in timestamps]
    elif isinstance(packet.get("relative_timestamps_ms"), list) and len(packet["relative_timestamps_ms"]) == len(samples):
        base_time = int(packet.get("timestamp_ms") or packet["timestamp"])
        offsets = [int(value) for value in packet["relative_timestamps_ms"]]
        sample_times = [base_time + offset for offset in offsets]
    else:
        end_time = int(packet.get("timestamp_ms") or packet["timestamp"])
        sample_times = [
            int(round(end_time - (len(samples) - 1 - index) * SAMPLE_PERIOD_MS))
            for index in range(len(samples))
        ]

    encoder_counts = packet.get("encoder_counts")
    if encoder_counts is None:
        encoder_counts = packet.get("encoder_count")
    calibration = float(packet.get("encoder_mm_per_count", 1.0))
    distances = packet.get("distance")
    source = canonical_distance_source(str(packet.get("distance_source", "Estimated")))
    sample_encoder_counts: list[float | None]

    if isinstance(encoder_counts, list) and len(encoder_counts) == len(samples):
        sample_encoder_counts = [float(value) for value in encoder_counts]
        sample_distances = [count * calibration for count in sample_encoder_counts]
        source = "Encoder"
    elif isinstance(distances, list) and len(distances) == len(samples):
        sample_distances = [float(value) for value in distances]
        sample_encoder_counts = [None] * len(samples)
        if encoder_counts is not None:
            source = "Encoder"
            sample_encoder_counts = [float(encoder_counts)] * len(samples)
    elif encoder_counts is not None:
        counts = float(encoder_counts)
        sample_distances = [counts * calibration] * len(samples)
        sample_encoder_counts = [counts] * len(samples)
        source = "Encoder"
    else:
        distance = float(packet.get("distance_mm", 0.0))
        sample_distances = [distance] * len(samples)
        sample_encoder_counts = [None] * len(samples)
        if source == "Encoder":
            source = "Estimated"

    return [
        {
            "timestamp_ms": sample_times[index],
            "voltage": samples[index],
            "encoder_count": sample_encoder_counts[index],
            "distance_mm": sample_distances[index],
            "distance_source": source,
        }
        for index in range(len(samples))
    ]
