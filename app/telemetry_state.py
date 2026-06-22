"""Thread-safe telemetry state for live ingest and polling APIs."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from threading import RLock
from typing import Any

from .features import WINDOW_SIZE, WINDOW_STRIDE


class TelemetryState:
    def __init__(self, history_size: int = 500, event_size: int = 100) -> None:
        self._lock = RLock()
        self._voltage_buffer: list[float] = []
        self._distance_buffer: list[float] = []
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._events: deque[dict[str, Any]] = deque(maxlen=event_size)
        self._latest_packet: dict[str, Any] | None = None
        self._latest_frame: dict[str, Any] | None = None
        self._latest_metrics: dict[str, Any] | None = None
        self._latest_timestamp: int | None = None

    def append_packet(self, packet: dict[str, Any]) -> list[tuple[list[float], float]]:
        with self._lock:
            self._latest_packet = deepcopy(packet)
            self._latest_timestamp = int(packet["timestamp"])
            self._history.append(deepcopy(packet))

            if not packet.get("arc_on", True):
                self._voltage_buffer.clear()
                self._distance_buffer.clear()
                return []

            samples = [float(value) for value in packet["voltage"]]
            distance = float(packet.get("distance_mm", 0.0))
            self._voltage_buffer.extend(samples)
            self._distance_buffer.extend([distance] * len(samples))

            windows: list[tuple[list[float], float]] = []
            while len(self._voltage_buffer) >= WINDOW_SIZE:
                windows.append((
                    list(self._voltage_buffer[:WINDOW_SIZE]),
                    float(self._distance_buffer[WINDOW_SIZE - 1]),
                ))
                del self._voltage_buffer[:WINDOW_STRIDE]
                del self._distance_buffer[:WINDOW_STRIDE]
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
