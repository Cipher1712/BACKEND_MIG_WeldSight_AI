"""Dataset loading for raw weld traces and precomputed feature tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import WindowFeatures, feature_matrix, windowize
from .physics import assess


def load_feature_dataset(path: str, require_labels: bool = True) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    elif source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    elif source.suffix.lower() == ".json":
        payload = json.loads(source.read_text())
        records = payload if isinstance(payload, list) else payload.get("welds", [])
        rows, labels, violations = [], [], []
        for record in records:
            voltage = record.get("voltage", [])
            distance = record.get("distance")
            for _, feature in windowize(voltage, distance):
                rows.append(feature)
                labels.append(record.get("label", "stable_arc"))
                violations.append(list(assess(
                    feature, record.get("material", "mild_steel"),
                    float(record.get("thickness_mm", 6.0)),
                ).violations.values()))
        return feature_matrix(rows), np.asarray(labels) if labels else None, np.asarray(violations, dtype=np.float32)
    else:
        raise ValueError("dataset must be CSV, Parquet, or JSON")

    missing = set(WindowFeatures.names()) - set(frame.columns)
    if missing:
        raise ValueError(f"missing feature columns: {sorted(missing)}")
    labels = frame["label"].astype(str).to_numpy() if "label" in frame else None
    if require_labels and labels is None:
        raise ValueError("classifier dataset requires a 'label' column")
    X = frame[WindowFeatures.names()].to_numpy(dtype=np.float32)
    violation_cols = ["physics_arc_stability", "physics_variance", "physics_short_circuit",
                      "physics_spectral_entropy"]
    violations = frame[violation_cols].to_numpy(dtype=np.float32) if set(violation_cols) <= set(frame.columns) else np.zeros((len(frame), 4), dtype=np.float32)
    return X, labels, violations
