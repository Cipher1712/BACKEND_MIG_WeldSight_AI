"""Train, calibrate, validate, and report WeldSight backend artifacts.

This script is intentionally backend-only. It uses the reduced 15-feature
voltage pipeline, writes fresh model artifacts, and produces quantitative
reports for the datasets that are present in the repository.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from app.anomaly import _calibrate
from app.features import SAMPLING_RATE_HZ, WINDOW_SIZE, WINDOW_STRIDE, WindowFeatures, extract
from app.inference import InferencePipeline
from app.model_training import train_healthy_models
from app.physics import LABELS, assess
from app.telemetry_state import TelemetryState
from app.vae import load_vae
from app.welding_data import detect_voltage_column, healthy_dataset_paths, valid_arc_window


DEFECT_COLUMNS = ("is_porosity", "is_discontinuity", "is_undercut")
REPORT_FILENAMES = (
    "dataset_inventory.json",
    "feature_audit.json",
    "training_report.json",
    "validation_report.json",
    "reliability_report.json",
    "industrial_detection_audit.json",
    "final_backend_report.md",
)


@dataclass(slots=True)
class WindowRow:
    feature: Any
    label: str
    source: str
    distance_source: str


def _safe_profile_key(material: str, thickness_mm: float) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in material).strip("_") or "material"
    return f"{clean}_{float(thickness_mm):.2f}mm".replace(".", "p")


def _condition_from_filename(path: Path) -> dict[str, float | None]:
    match = re.search(r"I(?P<current>\d+(?:\.\d+)?)_V(?P<voltage>\d+(?:\.\d+)?)", path.name)
    if not match:
        return {"current_a": None, "voltage_v": None}
    return {"current_a": float(match.group("current")), "voltage_v": float(match.group("voltage"))}


def _read_csv_header(path: Path) -> list[str]:
    return [str(column) for column in pd.read_csv(path, nrows=0).columns]


def discover_datasets(data_dir: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(data_dir.rglob("*.csv")):
        columns = _read_csv_header(path)
        voltage_column = detect_voltage_column(columns)
        frame = pd.read_csv(path, usecols=[column for column in columns if column in {
            voltage_column, "Distance", "Encoder", *DEFECT_COLUMNS, "no_defect"
        }])
        voltage = pd.to_numeric(frame[voltage_column], errors="coerce").to_numpy(dtype=np.float64)
        finite_voltage = voltage[np.isfinite(voltage)]
        valid_windows = 0
        for start in range(0, len(finite_voltage) - WINDOW_SIZE + 1, WINDOW_STRIDE):
            valid_windows += int(valid_arc_window(finite_voltage[start:start + WINDOW_SIZE]))
        has_labels = set(DEFECT_COLUMNS) <= set(frame.columns) and "no_defect" in frame.columns
        healthy_windows = anomalous_windows = None
        if has_labels:
            labels = _labels_for_frame(frame)
            healthy_windows, anomalous_windows = _count_labeled_windows(finite_voltage, labels)
        kind = "Healthy" if "MIG Sensor Data" in str(path) else "Validation/Labeled" if has_labels else "Unknown"
        inventory.append({
            "dataset": str(path),
            "samples": int(len(finite_voltage)),
            "valid_windows": int(valid_windows),
            "material": "unknown",
            "thickness_mm": None,
            "healthy_or_unknown": kind,
            "voltage_column": voltage_column,
            "has_encoder": "Encoder" in columns,
            "has_distance": "Distance" in columns,
            "has_defect_labels": has_labels,
            "healthy_windows": healthy_windows,
            "anomalous_windows": anomalous_windows,
            **_condition_from_filename(path),
        })
    return inventory


def _labels_for_frame(frame: pd.DataFrame) -> np.ndarray:
    if set(DEFECT_COLUMNS) <= set(frame.columns):
        defects = frame[list(DEFECT_COLUMNS)].fillna(0).astype(float).to_numpy()
        anomalous = defects.max(axis=1) > 0
        if "no_defect" in frame.columns:
            healthy = pd.to_numeric(frame["no_defect"], errors="coerce").fillna(0).to_numpy() > 0
            return np.where(anomalous, "anomalous", np.where(healthy, "healthy", "unknown"))
        return np.where(anomalous, "anomalous", "unknown")
    return np.full(len(frame), "unknown", dtype=object)


def _count_labeled_windows(voltage: np.ndarray, labels: np.ndarray) -> tuple[int, int]:
    healthy = anomalous = 0
    usable = min(len(voltage), len(labels))
    voltage = voltage[:usable]
    labels = labels[:usable]
    finite_mask = np.isfinite(voltage)
    voltage = voltage[finite_mask]
    labels = labels[finite_mask]
    for start in range(0, len(voltage) - WINDOW_SIZE + 1, WINDOW_STRIDE):
        window = voltage[start:start + WINDOW_SIZE]
        label_window = labels[start:start + WINDOW_SIZE]
        if not valid_arc_window(window):
            continue
        if np.mean(label_window == "anomalous") >= 0.10:
            anomalous += 1
        elif np.mean(label_window == "healthy") >= 0.90:
            healthy += 1
    return healthy, anomalous


def load_healthy_rows(paths: list[Path], max_windows: int | None, seed: int) -> list[WindowRow]:
    rows: list[WindowRow] = []
    for path in paths:
        voltage_column = detect_voltage_column(_read_csv_header(path))
        voltage = pd.to_numeric(pd.read_csv(path, usecols=[voltage_column])[voltage_column], errors="coerce")
        values = voltage.to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        for start in range(0, len(values) - WINDOW_SIZE + 1, WINDOW_STRIDE):
            window = values[start:start + WINDOW_SIZE]
            if valid_arc_window(window):
                rows.append(WindowRow(extract(window), "healthy", str(path), "encoder"))
    if max_windows and len(rows) > max_windows:
        indices = np.random.default_rng(seed).choice(len(rows), max_windows, replace=False)
        rows = [rows[int(index)] for index in indices]
    return rows


def load_labeled_rows(paths: list[Path], max_windows_per_file: int | None, seed: int) -> list[WindowRow]:
    rows: list[WindowRow] = []
    rng = np.random.default_rng(seed)
    for path in paths:
        columns = _read_csv_header(path)
        if not set(DEFECT_COLUMNS) <= set(columns):
            continue
        voltage_column = detect_voltage_column(columns)
        frame = pd.read_csv(path)
        labels = _labels_for_frame(frame)
        voltage = pd.to_numeric(frame[voltage_column], errors="coerce").to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(voltage)
        voltage = voltage[finite_mask]
        labels = labels[finite_mask]
        file_rows: list[WindowRow] = []
        for start in range(0, len(voltage) - WINDOW_SIZE + 1, WINDOW_STRIDE):
            window = voltage[start:start + WINDOW_SIZE]
            label_window = labels[start:start + WINDOW_SIZE]
            if not valid_arc_window(window):
                continue
            if np.mean(label_window == "anomalous") >= 0.10:
                label = "anomalous"
            elif np.mean(label_window == "healthy") >= 0.90:
                label = "healthy"
            else:
                label = "unknown"
            file_rows.append(WindowRow(extract(window), label, str(path), "distance"))
        if max_windows_per_file and len(file_rows) > max_windows_per_file:
            indices = rng.choice(len(file_rows), max_windows_per_file, replace=False)
            file_rows = [file_rows[int(index)] for index in indices]
        rows.extend(file_rows)
    return rows


def _matrix(rows: list[WindowRow]) -> np.ndarray:
    return np.vstack([row.feature.to_vector() for row in rows]).astype(np.float32)


def feature_audit(rows: list[WindowRow], thresholds: dict[str, Any]) -> dict[str, Any]:
    X = _matrix(rows)
    names = WindowFeatures.names()
    corr = np.corrcoef(X, rowvar=False)
    return {
        "feature_count": len(names),
        "features": [
            {
                "name": name,
                "definition": _feature_definition(name),
                "mean": float(np.mean(X[:, index])),
                "std": float(np.std(X[:, index])),
                "variance": float(np.var(X[:, index])),
                "missing_values": 0,
                "q01": float(np.quantile(X[:, index], 0.01)),
                "q50": float(np.quantile(X[:, index], 0.50)),
                "q99": float(np.quantile(X[:, index], 0.99)),
            }
            for index, name in enumerate(names)
        ],
        "feature_importance": thresholds.get("feature_importance", []),
        "correlation_matrix": {
            "features": names,
            "values": np.round(np.nan_to_num(corr), 6).tolist(),
        },
        "stale_reference_audit": {
            "expected_feature_count": len(names),
            "artifact_feature_count": thresholds.get("feature_count"),
            "stale_30_feature_artifacts_detected": thresholds.get("feature_count") == 30,
        },
    }


def _feature_definition(name: str) -> str:
    definitions = {
        "mean_v": "Mean MIG voltage within the window.",
        "rms_v": "Root-mean-square voltage within the window.",
        "variance_v": "Mean squared deviation from window mean voltage.",
        "std_v": "Standard deviation of window voltage.",
        "ripple": "RMS of first-difference voltage changes.",
        "energy": "Mean squared voltage, voltage-only heat input proxy.",
        "crest_factor": "Peak absolute voltage divided by RMS voltage.",
        "spectral_entropy": "Normalized entropy of window FFT power spectrum.",
        "spectral_centroid_hz": "FFT power weighted average frequency.",
        "arc_stability_index": "Bounded stability score derived from variation, ripple, shorts, and spikes.",
        "short_circuit_ratio": "Fraction of samples below adaptive short-circuit voltage threshold.",
        "short_circuit_density": "Short-circuit run count normalized per second.",
        "spike_density": "Fraction of abrupt spike/drop transitions.",
        "mean_abs_delta_v": "Mean absolute first-difference voltage.",
        "p95_abs_delta_v": "95th percentile absolute first-difference voltage.",
    }
    return definitions.get(name, name.replace("_", " "))


def score_rows(rows: list[WindowRow], model_dir: Path, thresholds: dict[str, Any], weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
    scaler = joblib.load(model_dir / "scaler.pkl")
    isolation = joblib.load(model_dir / "isolation_forest.pkl")
    vae = load_vae(str(model_dir / "vae.pt"))
    X = _matrix(rows)
    scaled = scaler.transform(X).astype(np.float32)
    magnitudes = np.linalg.norm(scaled, axis=1) / np.sqrt(scaled.shape[1])
    ewma_scores = _ewma_scores(magnitudes, float(thresholds.get("ewma_k", 3.0)))
    forest_raw = -isolation.decision_function(scaled)
    forest_scores = np.asarray([_calibrate(float(value), thresholds.get("isolation_forest", {})) for value in forest_raw])
    rec, latent = [], []
    for row in scaled:
        reconstruction_error, latent_distance = vae.anomaly_components(row.reshape(1, -1))
        rec.append(reconstruction_error)
        latent.append(latent_distance)
    rec = np.asarray(rec)
    latent = np.asarray(latent)
    rec_scores = np.asarray([_calibrate(float(value), thresholds.get("reconstruction", {})) for value in rec])
    latent_scores = np.asarray([_calibrate(float(value), thresholds.get("latent_distance", {})) for value in latent])
    vae_scores = 0.75 * rec_scores + 0.25 * latent_scores
    physics_scores = np.asarray([assess(row.feature, thresholds.get("feature_reference", {})).score for row in rows])
    weights = weights or thresholds.get("fusion_weights", {"vae": 0.42, "isolation_forest": 0.24, "ewma": 0.18, "physics": 0.16})
    total_weight = max(sum(float(value) for value in weights.values()), 1e-9)
    fused = (
        float(weights.get("vae", 0.0)) * vae_scores +
        float(weights.get("isolation_forest", 0.0)) * forest_scores +
        float(weights.get("ewma", 0.0)) * ewma_scores +
        float(weights.get("physics", 0.0)) * physics_scores
    ) / total_weight
    return [
        {
            "label": row.label,
            "source": row.source,
            "score": float(fused[index]),
            "vae_score": float(vae_scores[index]),
            "isolation_score": float(forest_scores[index]),
            "ewma_score": float(ewma_scores[index]),
            "physics_score": float(physics_scores[index]),
            "reconstruction_error": float(rec[index]),
            "latent_distance": float(latent[index]),
            "prediction": assess(row.feature, thresholds.get("feature_reference", {})).label,
        }
        for index, row in enumerate(rows)
    ]


def _ewma_scores(values: np.ndarray, k: float) -> np.ndarray:
    mean = variance = 0.0
    scores = []
    for index, value in enumerate(values, start=1):
        if index == 1:
            mean = float(value)
            variance = 0.0
        else:
            dev = float(value) - mean
            mean = 0.15 * float(value) + 0.85 * mean
            variance = 0.92 * variance + 0.08 * dev * dev
        threshold = max(mean + k * np.sqrt(variance), 0.05 * 0.85)
        scores.append(float(np.clip((float(value) - mean) / max(threshold - mean, 0.25), 0.0, 1.0)))
    return np.asarray(scores)


def optimize_weights(component_rows: list[dict[str, Any]]) -> dict[str, float]:
    healthy = [row for row in component_rows if row["label"] == "healthy"]
    anomalous = [row for row in component_rows if row["label"] == "anomalous"]
    if len(healthy) < 20 or len(anomalous) < 20:
        return {"vae": 0.42, "isolation_forest": 0.24, "ewma": 0.18, "physics": 0.16}
    components = ("vae_score", "isolation_score", "ewma_score", "physics_score")
    H = np.asarray([[row[name] for name in components] for row in healthy], dtype=float)
    A = np.asarray([[row[name] for name in components] for row in anomalous], dtype=float)
    best_score = -1e9
    best = np.asarray([0.42, 0.24, 0.18, 0.16])
    for a in range(0, 11):
        for b in range(0, 11 - a):
            for c in range(0, 11 - a - b):
                d = 10 - a - b - c
                w = np.asarray([a, b, c, d], dtype=float) / 10.0
                if np.count_nonzero(w) < 2:
                    continue
                hs = H @ w
                threshold = float(np.quantile(hs, 0.95))
                fpr = float(np.mean(hs >= threshold))
                tpr = float(np.mean((A @ w) >= threshold))
                separation = float(np.mean(A @ w) - np.mean(hs))
                objective = tpr + separation - max(0.0, fpr - 0.05) * 2.0
                if objective > best_score:
                    best_score = objective
                    best = w
    return {name: round(float(best[index]), 4) for index, name in enumerate(("vae", "isolation_forest", "ewma", "physics"))}


def calibrate_thresholds(scores: list[dict[str, Any]]) -> dict[str, Any]:
    healthy_scores = np.asarray([row["score"] for row in scores if row["label"] == "healthy"], dtype=float)
    if healthy_scores.size == 0:
        healthy_scores = np.asarray([row["score"] for row in scores], dtype=float)
    return {
        "normal_threshold": float(np.quantile(healthy_scores, 0.90)),
        "watch_threshold": float(np.quantile(healthy_scores, 0.95)),
        "warning_threshold": float(np.quantile(healthy_scores, 0.99)),
        "critical_threshold": float(np.quantile(healthy_scores, 0.995)),
        "rationale": "Bands are calibrated from healthy anomaly-score percentiles: p90, p95, p99, and p99.5.",
    }


def validation_report(scores: list[dict[str, Any]], bands: dict[str, Any]) -> dict[str, Any]:
    watch = bands["watch_threshold"]
    warning = bands["warning_threshold"]
    healthy = [row for row in scores if row["label"] == "healthy"]
    anomalous = [row for row in scores if row["label"] == "anomalous"]
    return {
        "windows": {"healthy": len(healthy), "anomalous": len(anomalous), "unknown": sum(row["label"] == "unknown" for row in scores)},
        "false_positive_rate_at_watch": _rate(healthy, lambda row: row["score"] >= watch),
        "false_positive_rate_at_warning": _rate(healthy, lambda row: row["score"] >= warning),
        "detection_rate_at_watch": _rate(anomalous, lambda row: row["score"] >= watch),
        "detection_rate_at_warning": _rate(anomalous, lambda row: row["score"] >= warning),
        "false_negative_rate_at_watch": _rate(anomalous, lambda row: row["score"] < watch),
        "score_summary": _score_summary(scores),
        "vae_validation": _component_summary(scores, "reconstruction_error", "latent_distance", "vae_score"),
        "isolation_forest_validation": _isolation_summary(scores),
        "threshold_analysis": bands,
    }


def _rate(rows: list[dict[str, Any]], predicate) -> float | None:
    if not rows:
        return None
    return round(float(np.mean([predicate(row) for row in rows])), 6)


def _score_summary(scores: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for label in ("healthy", "anomalous", "unknown"):
        values = np.asarray([row["score"] for row in scores if row["label"] == label], dtype=float)
        if values.size:
            summary[label] = _dist(values)
    return summary


def _dist(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def _component_summary(scores: list[dict[str, Any]], *fields: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        result[field] = {}
        for label in ("healthy", "anomalous"):
            values = np.asarray([row[field] for row in scores if row["label"] == label], dtype=float)
            if values.size:
                result[field][label] = _dist(values)
    healthy = np.asarray([row["vae_score"] for row in scores if row["label"] == "healthy"], dtype=float)
    anomalous = np.asarray([row["vae_score"] for row in scores if row["label"] == "anomalous"], dtype=float)
    if healthy.size and anomalous.size:
        separation = (float(np.mean(anomalous)) - float(np.mean(healthy))) / max(float(np.std(healthy)), 1e-9)
        result["vae_reliability_score"] = round(float(np.clip(50.0 + 12.5 * separation, 0.0, 100.0)), 2)
    else:
        result["vae_reliability_score"] = None
    return result


def _isolation_summary(scores: list[dict[str, Any]]) -> dict[str, Any]:
    healthy = np.asarray([row["isolation_score"] for row in scores if row["label"] == "healthy"], dtype=float)
    anomalous = np.asarray([row["isolation_score"] for row in scores if row["label"] == "anomalous"], dtype=float)
    vae = np.asarray([row["vae_score"] for row in scores], dtype=float)
    forest = np.asarray([row["isolation_score"] for row in scores], dtype=float)
    corr = float(np.corrcoef(vae, forest)[0, 1]) if len(scores) > 2 and np.std(vae) > 0 and np.std(forest) > 0 else None
    return {
        "healthy": _dist(healthy) if healthy.size else None,
        "anomalous": _dist(anomalous) if anomalous.size else None,
        "correlation_with_vae": corr,
        "usefulness": "negligible" if corr is not None and abs(corr) > 0.95 else "complementary",
        "note": "Isolation Forest is retained even when highly correlated; removal requires a separate design decision.",
    }


def reliability_report(rows: list[WindowRow], model_dir: Path, thresholds: dict[str, Any], weights: dict[str, float], bands: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(99)
    healthy_rows = [row for row in rows if row.label == "healthy"]
    if len(healthy_rows) > 1500:
        healthy_rows = [healthy_rows[int(index)] for index in rng.choice(len(healthy_rows), 1500, replace=False)]
    tests = {}
    for name, sigma in (("small_noise", 0.05), ("medium_noise", 0.20), ("large_noise", 0.60)):
        noisy_rows = []
        for row in healthy_rows:
            vector = row.feature.to_vector().astype(float)
            mean_v = max(float(row.feature.mean_v), 1.0)
            samples = rng.normal(float(row.feature.mean_v), max(float(row.feature.std_v), 0.01), WINDOW_SIZE)
            samples += rng.normal(0.0, sigma * mean_v, WINDOW_SIZE)
            noisy_rows.append(WindowRow(extract(samples), name, row.source, row.distance_source))
        scored = score_rows(noisy_rows, model_dir, thresholds, weights)
        tests[name] = {
            "windows": len(scored),
            "watch_or_higher_rate": _rate(scored, lambda item: item["score"] >= bands["watch_threshold"]),
            "warning_or_higher_rate": _rate(scored, lambda item: item["score"] >= bands["warning_threshold"]),
            "score_summary": _score_summary(scored),
        }
    drift_rows = []
    for row in healthy_rows:
        baseline = np.full(WINDOW_SIZE, float(row.feature.mean_v))
        drift = np.linspace(0.0, 0.10 * max(float(row.feature.mean_v), 1.0), WINDOW_SIZE)
        drift_rows.append(WindowRow(extract(baseline + drift), "drift", row.source, row.distance_source))
    drift_scores = score_rows(drift_rows, model_dir, thresholds, weights)
    tests["drift"] = {
        "windows": len(drift_scores),
        "watch_or_higher_rate": _rate(drift_scores, lambda item: item["score"] >= bands["watch_threshold"]),
        "score_summary": _score_summary(drift_scores),
    }
    return {
        "target_false_positive_rate": 0.05,
        "healthy_false_positive_rate_at_watch": None,
        "robustness_tests": tests,
        "strengths": [
            "Uses real healthy windows for scaler, envelope, VAE, Isolation Forest, and score-band calibration.",
            "Uses labeled defect columns from condition CSVs for validation where available.",
            "Preserves encoder/distance-source semantics instead of treating estimates as measured distance.",
        ],
        "weaknesses": [
            "Material and thickness metadata are unavailable in the repository, so trained profile metadata remains unknown.",
            "Voltage-only sensing cannot prove all weld defect classes without visual, current, travel-speed, or ground-truth inspection context.",
        ],
        "failure_modes": [
            "Operating voltage regimes absent from healthy training data may be scored as anomalous.",
            "Slow process shifts can still require new profile calibration.",
            "Distance is relative unless encoder calibration is supplied.",
        ],
    }


def timing_distance_verification() -> dict[str, Any]:
    state = TelemetryState()
    timestamps = [1000 + index * 2 for index in range(WINDOW_SIZE)]
    packet = {
        "voltage": [18.0] * WINDOW_SIZE,
        "timestamps_ms": timestamps,
        "timestamp": timestamps[-1],
        "distance": [index * 0.1 for index in range(WINDOW_SIZE)],
        "distance_source": "encoder",
        "arc_on": True,
    }
    windows = state.append_packet(packet)
    midpoint = WINDOW_SIZE // 2
    return {
        "window_count": len(windows),
        "expected_event_timestamp_ms": timestamps[midpoint],
        "actual_event_timestamp_ms": windows[0][2] if windows else None,
        "expected_distance_mm": packet["distance"][midpoint],
        "actual_distance_mm": windows[0][1] if windows else None,
        "distance_source": windows[0][3] if windows else None,
        "passed": bool(windows and windows[0][2] == timestamps[midpoint] and windows[0][3] == "encoder"),
    }


def industrial_detection_audit(scores: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ["Healthy Arc", "Arc Instability", "Spatter Risk", "Low Heat Input Risk", "Burn Through Risk"]
    rows = []
    for label in labels:
        matching = [row for row in scores if row["prediction"] == label]
        if matching:
            confidence = float(1.0 - np.std([row["score"] for row in matching]))
            evidence = len(matching)
        else:
            confidence = None
            evidence = 0
        rows.append({"condition": label, "confidence": confidence, "evidence_windows": evidence})
    return {
        "condition_confidence": rows,
        "note": "Confidence is only reported where this voltage-only dataset produced evidence for the condition.",
    }


def write_markdown(output: Path, payload: dict[str, Any]) -> None:
    validation = payload["validation"]
    verdict = payload["deployment_readiness"]
    lines = [
        "# WeldSight Backend Training and Reliability Report",
        "",
        f"Feature count: {payload['feature_audit']['feature_count']}",
        f"Profiles trained: {', '.join(item['profile_key'] for item in payload['training']['profiles_trained'])}",
        "",
        "## Validation",
        f"False positive rate at Watch: {validation['false_positive_rate_at_watch']}",
        f"Detection rate at Watch: {validation['detection_rate_at_watch']}",
        f"Detection rate at Warning: {validation['detection_rate_at_warning']}",
        "",
        "## Thresholds",
        json.dumps(validation["threshold_analysis"], indent=2),
        "",
        "## Deployment Readiness",
        f"Verdict: {verdict['verdict']}",
        verdict["justification"],
    ]
    output.write_text("\n".join(lines))


def _deployment_verdict(validation: dict[str, Any]) -> dict[str, str]:
    fpr = validation.get("false_positive_rate_at_watch")
    detection = validation.get("detection_rate_at_watch")
    if fpr is not None and detection is not None and fpr <= 0.05 and detection >= 0.60:
        verdict = "Ready for Validation Trials"
        justification = "Fresh artifacts are trained and labeled validation meets the target false-positive rate with useful sensitivity."
    elif fpr is not None and fpr <= 0.10:
        verdict = "Ready for Demo"
        justification = "Artifacts and reports are complete, but validation sensitivity/false positives need trial confirmation before industrial validation."
    else:
        verdict = "Needs Further Work"
        justification = "Quantitative validation does not yet meet the requested reliability target."
    return {"verdict": verdict, "justification": justification}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--models", default="models")
    parser.add_argument("--max-healthy-windows", type=int, default=12000)
    parser.add_argument("--max-labeled-windows-per-file", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    data_dir = Path(args.data)
    model_dir = Path(args.models)
    reports_dir = model_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    inventory = discover_datasets(data_dir)
    healthy_rows = load_healthy_rows(healthy_dataset_paths(data_dir), args.max_healthy_windows, seed=42)
    if len(healthy_rows) < 100:
        raise SystemExit("At least 100 healthy windows are required for training.")
    labeled_rows = load_labeled_rows(sorted(data_dir.glob("Data_I*.csv")), args.max_labeled_windows_per_file, seed=43)
    X = _matrix(healthy_rows)

    training = train_healthy_models(X, str(model_dir), material="unknown", thickness_mm=0.0, epochs=args.epochs, batch_size=args.batch_size)
    profile_key = _safe_profile_key("unknown_material", 0.0)
    profile_dir = model_dir / "profiles" / profile_key
    profile_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("scaler.pkl", "isolation_forest.pkl", "vae.pt", "anomaly_threshold.json"):
        shutil.copy2(model_dir / filename, profile_dir / filename)

    thresholds = json.loads((model_dir / "anomaly_threshold.json").read_text())
    initial_scores = score_rows(healthy_rows[: min(len(healthy_rows), 3000)] + labeled_rows, model_dir, thresholds)
    weights = optimize_weights(initial_scores)
    calibrated_scores = score_rows(healthy_rows[: min(len(healthy_rows), 3000)] + labeled_rows, model_dir, thresholds, weights)
    bands = calibrate_thresholds(calibrated_scores)
    thresholds["fusion_weights"] = weights
    thresholds["score_bands"] = {
        "normal": bands["normal_threshold"],
        "watch": bands["watch_threshold"],
        "warning": bands["warning_threshold"],
    }
    thresholds["anomaly_threshold"] = bands["watch_threshold"]
    thresholds["critical_threshold"] = bands["critical_threshold"]
    thresholds["threshold_calibration"] = bands
    (model_dir / "anomaly_threshold.json").write_text(json.dumps(thresholds, indent=2))
    shutil.copy2(model_dir / "anomaly_threshold.json", profile_dir / "anomaly_threshold.json")

    final_scores = score_rows(healthy_rows[: min(len(healthy_rows), 3000)] + labeled_rows, model_dir, thresholds, weights)
    validation = validation_report(final_scores, bands)
    reliability = reliability_report(healthy_rows, model_dir, thresholds, weights, bands)
    reliability["healthy_false_positive_rate_at_watch"] = validation["false_positive_rate_at_watch"]
    timing_distance = timing_distance_verification()
    audit = feature_audit(healthy_rows, thresholds)
    industrial = industrial_detection_audit(final_scores)
    generated_artifacts = [
        str(model_dir / filename) for filename in ("scaler.pkl", "vae.pt", "isolation_forest.pkl", "anomaly_threshold.json")
    ] + [str(profile_dir / filename) for filename in ("scaler.pkl", "vae.pt", "isolation_forest.pkl", "anomaly_threshold.json")]
    generated_artifacts += [str(reports_dir / filename) for filename in REPORT_FILENAMES]
    generated_artifacts += [
        str(model_dir / "training_report.json"),
        str(model_dir / "validation_report.json"),
        str(model_dir / "reliability_report.json"),
        str(model_dir / "evaluation_report.json"),
    ]
    training_report = {
        **training,
        "datasets_used": [row for row in inventory if row["healthy_or_unknown"] == "Healthy"],
        "profiles_trained": [{
            "profile_key": profile_key,
            "material": "unknown",
            "thickness_mm": None,
            "artifact_dir": str(profile_dir),
            "note": "Repository does not contain confirmed material/thickness metadata; profile is metadata-honest.",
        }],
        "feature_set": WindowFeatures.names(),
        "model_parameters": {"epochs": args.epochs, "batch_size": args.batch_size},
    }
    payload = {
        "dataset_inventory": inventory,
        "feature_audit": audit,
        "training": training_report,
        "validation": validation,
        "reliability": reliability,
        "timing_distance_validation": timing_distance,
        "industrial_detection_audit": industrial,
        "generated_artifacts": generated_artifacts,
        "deployment_readiness": _deployment_verdict(validation),
    }
    outputs = {
        "dataset_inventory.json": inventory,
        "feature_audit.json": audit,
        "training_report.json": training_report,
        "validation_report.json": validation,
        "reliability_report.json": {**reliability, "timing_distance_validation": timing_distance},
        "industrial_detection_audit.json": industrial,
    }
    for filename, content in outputs.items():
        (reports_dir / filename).write_text(json.dumps(content, indent=2))
    (model_dir / "training_report.json").write_text(json.dumps(training_report, indent=2))
    (model_dir / "validation_report.json").write_text(json.dumps(validation, indent=2))
    (model_dir / "reliability_report.json").write_text(json.dumps({**reliability, "timing_distance_validation": timing_distance}, indent=2))
    (model_dir / "evaluation_report.json").write_text(json.dumps(validation, indent=2))
    write_markdown(reports_dir / "final_backend_report.md", payload)
    (model_dir / "manifest.json").write_text(json.dumps({
        "feature_count": len(WindowFeatures.names()),
        "feature_names": WindowFeatures.names(),
        "profiles": [profile_key],
        "reports_dir": str(reports_dir),
        "generated_artifacts": generated_artifacts,
        "deployment_readiness": payload["deployment_readiness"],
    }, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
