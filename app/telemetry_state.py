"""Thread-safe telemetry state for live ingest and polling APIs."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from threading import RLock
from typing import Any

from .features import WINDOW_SIZE, WINDOW_STRIDE


SAMPLE_PERIOD_MS = 1000.0 / 750.0


class TelemetryState:
    def __init__(self, history_size: int = 500, event_size: int = 100) -> None:
        self._lock = RLock()
        self._voltage_buffer: list[float] = []
        self._distance_buffer: list[float] = []
        self._timestamp_buffer: list[int] = []
        self._distance_source_buffer: list[str] = []
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._events: deque[dict[str, Any]] = deque(maxlen=event_size)
        self._latest_packet: dict[str, Any] | None = None
        self._latest_frame: dict[str, Any] | None = None
        self._latest_metrics: dict[str, Any] | None = None
        self._latest_timestamp: int | None = None

    def append_packet(self, packet: dict[str, Any]) -> list[tuple[list[float], float, int, str]]:
        with self._lock:
            self._latest_packet = deepcopy(packet)
            self._latest_timestamp = int(packet["timestamp"])
            self._history.append(deepcopy(packet))

            if not packet.get("arc_on", True):
                self._voltage_buffer.clear()
                self._distance_buffer.clear()
                self._timestamp_buffer.clear()
                self._distance_source_buffer.clear()
                return []

            samples = [float(value) for value in packet["voltage"]]
            timestamps = packet.get("timestamps_ms")
            if isinstance(timestamps, list) and len(timestamps) == len(samples):
                sample_times = [int(value) for value in timestamps]
            else:
                end_time = int(packet.get("timestamp_ms") or packet["timestamp"])
                sample_times = [
                    int(round(end_time - (len(samples) - 1 - index) * SAMPLE_PERIOD_MS))
                    for index in range(len(samples))
                ]

            distances = packet.get("distance")
            distance_source = str(packet.get("distance_source", "estimated"))
            if isinstance(distances, list) and len(distances) == len(samples):
                sample_distances = [float(value) for value in distances]
                distance_source = "encoder" if packet.get("encoder_counts") is not None else distance_source
            elif packet.get("encoder_counts") is not None:
                counts = float(packet["encoder_counts"])
                calibration = float(packet.get("encoder_mm_per_count", 1.0))
                sample_distances = [counts * calibration] * len(samples)
                distance_source = "encoder"
            else:
                distance = float(packet.get("distance_mm", 0.0))
                sample_distances = [distance] * len(samples)
            self._voltage_buffer.extend(samples)
            self._distance_buffer.extend(sample_distances)
            self._timestamp_buffer.extend(sample_times)
            self._distance_source_buffer.extend([distance_source] * len(samples))

            windows: list[tuple[list[float], float, int, str]] = []
            while len(self._voltage_buffer) >= WINDOW_SIZE:
                midpoint = WINDOW_SIZE // 2
                windows.append((
                    list(self._voltage_buffer[:WINDOW_SIZE]),
                    float(self._distance_buffer[midpoint]),
                    int(self._timestamp_buffer[midpoint]),
                    str(self._distance_source_buffer[midpoint]),
                ))
                del self._voltage_buffer[:WINDOW_STRIDE]
                del self._distance_buffer[:WINDOW_STRIDE]
                del self._timestamp_buffer[:WINDOW_STRIDE]
                del self._distance_source_buffer[:WINDOW_STRIDE]
            return windows

    def record_frame(self, frame: dict[str, Any]) -> None:
        metrics = {
            "quality_index": frame.get("quality_index"),
            "quality_score": frame.get("quality_score"),
            "stability": frame.get("stability_score"),
            "anomalies": {
                "detected": frame.get("anomaly_detected"),
                "score": frame.get("anomaly_score"),
                "threshold": frame.get("anomaly_threshold"),
                "severity": frame.get("severity"),
                "physics_label": frame.get("physics_label"),
                "ml_label": frame.get("ml_label"),
            },
            "status": frame.get("status"),
            "diagnosis": frame.get("diagnosis"),
            "model_ready": frame.get("model_ready"),
            "timestamp": frame.get("timestamp"),
        }
        with self._lock:
            self._latest_frame = deepcopy(frame)
            self._latest_metrics = deepcopy(metrics)
            self._latest_timestamp = int(frame["timestamp"])
            if frame.get("anomaly_detected"):
                self._events.append(deepcopy(frame))

    def latest_telemetry(self) -> dict[str, Any]:
        with self._lock:
            if self._latest_packet is None:
                return {}
            packet = deepcopy(self._latest_packet)
            if self._latest_frame is not None:
                packet["latest_inference"] = deepcopy(self._latest_frame)
            return packet

    def latest_metrics(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._latest_metrics or {})

    def latest_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(event) for event in reversed(self._events)]

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(packet) for packet in self._history]


telemetry_state = TelemetryState()
