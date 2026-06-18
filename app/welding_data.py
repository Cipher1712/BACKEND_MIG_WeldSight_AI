"""Read and validate real MIG voltage datasets without using other channels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .features import WINDOW_SIZE, WINDOW_STRIDE, WindowFeatures, extract

VOLTAGE_ALIASES = {
    "migvoltage", "mig voltage", "voltage", "arcvoltage", "voltage_v",
    "migvolatge",  # observed typo in data/MIG Sensor Data
}


@dataclass(slots=True)
class DatasetSummary:
    path: str
    voltage_column: str
    samples: int
    valid_windows: int


def detect_voltage_column(columns: list[str]) -> str:
    for column in columns:
        if str(column).strip().lower() in VOLTAGE_ALIASES:
            return str(column)
    raise ValueError(f"no MIG voltage column found; columns={columns}")


def read_voltage(path: str | Path) -> np.ndarray:
    source = Path(path)
    column = detect_voltage_column(list(pd.read_csv(source, nrows=0).columns))
    values = pd.to_numeric(pd.read_csv(source, usecols=[column])[column], errors="coerce")
    voltage = values.to_numpy(dtype=np.float64)
    return voltage[np.isfinite(voltage)]


def valid_arc_window(window: np.ndarray) -> bool:
    """Reject arc-off, corrupt, and physically impossible windows."""
    if len(window) != WINDOW_SIZE or not np.isfinite(window).all():
        return False
    if np.median(window) < 8.0:
        return False
    if np.mean((window >= -2.0) & (window <= 120.0)) < 0.99:
        return False
    if np.mean(window > 2.0) < 0.75:
        return False
    return True


def iter_real_windows(path: str | Path) -> Iterator[np.ndarray]:
    voltage = read_voltage(path)
    for start in range(0, len(voltage) - WINDOW_SIZE + 1, WINDOW_STRIDE):
        window = voltage[start:start + WINDOW_SIZE]
        if valid_arc_window(window):
            yield window


def load_real_feature_matrix(
    paths: list[Path], max_windows: int | None = None, seed: int = 42,
) -> tuple[np.ndarray, list[DatasetSummary]]:
    rows: list[np.ndarray] = []
    summaries: list[DatasetSummary] = []
    for path in paths:
        column = detect_voltage_column(list(pd.read_csv(path, nrows=0).columns))
        voltage = read_voltage(path)
        count = 0
        for start in range(0, len(voltage) - WINDOW_SIZE + 1, WINDOW_STRIDE):
            window = voltage[start:start + WINDOW_SIZE]
            if valid_arc_window(window):
                rows.append(extract(window).to_vector())
                count += 1
        summaries.append(DatasetSummary(str(path), column, len(voltage), count))
    if not rows:
        raise ValueError("no valid real welding windows found")
    matrix = np.asarray(rows, dtype=np.float32)
    if max_windows and len(matrix) > max_windows:
        indices = np.random.default_rng(seed).choice(len(matrix), max_windows, replace=False)
        matrix = matrix[indices]
    return matrix, summaries


def healthy_dataset_paths(data_dir: str | Path = "data") -> list[Path]:
    return sorted((Path(data_dir) / "MIG Sensor Data").glob("*.csv"))


def condition_dataset_paths(data_dir: str | Path = "data") -> list[Path]:
    return sorted(Path(data_dir).glob("Data_I*.csv"))
