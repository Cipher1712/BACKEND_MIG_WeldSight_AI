"""Generate anomaly score separation plots and bottleneck analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from app.anomaly import _calibrate
from app.physics import assess
from app.vae import load_vae
from scripts.complete_backend_workflow import (
    _matrix,
    _score_summary,
    load_healthy_rows,
    load_labeled_rows,
    optimize_weights,
    score_rows,
)
from app.welding_data import healthy_dataset_paths


COMPONENTS = {
    "fused_anomaly": "score",
    "vae_score": "vae_score",
    "vae_reconstruction": "reconstruction_error",
    "isolation_forest": "isolation_score",
    "physics": "physics_score",
}


def _overlap_coefficient(healthy: np.ndarray, anomalous: np.ndarray, bins: int = 120) -> float:
    if healthy.size == 0 or anomalous.size == 0:
        return float("nan")
    lo = float(min(np.min(healthy), np.min(anomalous)))
    hi = float(max(np.max(healthy), np.max(anomalous)))
    if hi <= lo:
        return 1.0
    h_density, edges = np.histogram(healthy, bins=bins, range=(lo, hi), density=True)
    a_density, _ = np.histogram(anomalous, bins=edges, density=True)
    widths = np.diff(edges)
    return float(np.sum(np.minimum(h_density, a_density) * widths))


def _directional_values(healthy: np.ndarray, anomalous: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if np.mean(anomalous) >= np.mean(healthy):
        return healthy, anomalous, "higher_is_more_anomalous"
    return -healthy, -anomalous, "lower_is_more_anomalous"


def _max_detection_metrics(healthy: np.ndarray, anomalous: np.ndarray) -> dict[str, Any]:
    h, a, direction = _directional_values(healthy, anomalous)
    y_true = np.concatenate([np.zeros_like(h), np.ones_like(a)])
    y_score = np.concatenate([h, a])
    if np.unique(y_true).size < 2 or np.unique(y_score).size < 2:
        return {"auc": None, "direction": direction}
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = float(roc_auc_score(y_true, y_score))
    youden_index = tpr - fpr
    best_youden_idx = int(np.argmax(youden_index))
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_f1_idx = int(np.argmax(f1))
    threshold_for_f1 = float(pr_thresholds[min(best_f1_idx, len(pr_thresholds) - 1)]) if len(pr_thresholds) else None
    predictions = y_score >= (threshold_for_f1 if threshold_for_f1 is not None else roc_thresholds[best_youden_idx])
    tp = int(np.sum((predictions == 1) & (y_true == 1)))
    fp = int(np.sum((predictions == 1) & (y_true == 0)))
    tn = int(np.sum((predictions == 0) & (y_true == 0)))
    fn = int(np.sum((predictions == 0) & (y_true == 1)))
    specificity = tn / max(tn + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    return {
        "direction": direction,
        "auc": roc_auc,
        "pr_auc": float(auc(recall, precision)),
        "ks_statistic": float(np.max(np.abs(tpr - fpr))),
        "overlap_coefficient": _overlap_coefficient(healthy, anomalous),
        "best_youden": {
            "threshold": float(roc_thresholds[best_youden_idx]),
            "balanced_accuracy": float((tpr[best_youden_idx] + (1.0 - fpr[best_youden_idx])) / 2.0),
            "sensitivity": float(tpr[best_youden_idx]),
            "false_positive_rate": float(fpr[best_youden_idx]),
        },
        "best_f1": {
            "threshold": threshold_for_f1,
            "f1": float(f1[best_f1_idx]),
            "precision": float(precision[best_f1_idx]),
            "recall": float(recall[best_f1_idx]),
            "specificity": float(specificity),
            "sensitivity": float(sensitivity),
            "false_positive_rate": float(1.0 - specificity),
            "false_negative_rate": float(1.0 - sensitivity),
        },
    }


def _plot_distribution(name: str, healthy: np.ndarray, anomalous: np.ndarray, output: Path, threshold: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(healthy, bins=80, density=True, alpha=0.55, label=f"Healthy (n={len(healthy)})", color="#2563eb")
    ax.hist(anomalous, bins=80, density=True, alpha=0.55, label=f"Anomalous (n={len(anomalous)})", color="#dc2626")
    if threshold is not None:
        ax.axvline(threshold, color="#111827", linestyle="--", linewidth=1.5, label="Current Watch threshold")
    ax.set_title(name.replace("_", " ").title())
    ax.set_xlabel("Score / component value")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _bottleneck_verdict(metrics: dict[str, Any], feature_audit: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    fused_auc = metrics["fused_anomaly"]["auc"]
    best_auc = max(value.get("auc") or 0.0 for value in metrics.values())
    overlap = metrics["fused_anomaly"]["overlap_coefficient"]
    threshold_gap = metrics["fused_anomaly"]["best_youden"]["balanced_accuracy"] - metrics["fused_anomaly"]["best_f1"]["f1"]
    high_corr_pairs = []
    corr = np.asarray(feature_audit.get("correlation_matrix", {}).get("values", []), dtype=float)
    names = feature_audit.get("correlation_matrix", {}).get("features", [])
    if corr.size:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if abs(corr[i, j]) >= 0.98:
                    high_corr_pairs.append([names[i], names[j], float(corr[i, j])])

    if best_auc < 0.60 and overlap > 0.75:
        primary = "dataset/label separability"
        reason = "Healthy and anomalous labels occupy nearly the same voltage-only score distributions; even the best theoretical threshold is weak."
    elif fused_auc + 0.05 < best_auc:
        primary = "fusion/model weighting"
        reason = "At least one component separates better than the fused score, so fusion is leaving performance on the table."
    elif high_corr_pairs:
        primary = "feature redundancy"
        reason = "Several retained voltage features are almost perfectly correlated, limiting independent information."
    elif threshold_gap > 0.20:
        primary = "threshold operating point"
        reason = "The score has separation, but the chosen operating threshold is poorly aligned with desired precision/recall."
    else:
        primary = "model capacity/features"
        reason = "The dataset has some separability, but the current voltage-only model components are not extracting enough signal."
    return {
        "primary_bottleneck": primary,
        "reason": reason,
        "supporting_evidence": {
            "fused_auc": fused_auc,
            "best_component_auc": best_auc,
            "fused_overlap_coefficient": overlap,
            "current_fusion_weights": weights,
            "highly_correlated_feature_pairs_count": len(high_corr_pairs),
            "example_high_correlation_pairs": high_corr_pairs[:10],
        },
    }


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Anomaly Score Separation Report",
        "",
        "## Verdict",
        f"Primary bottleneck: **{report['bottleneck']['primary_bottleneck']}**",
        report["bottleneck"]["reason"],
        "",
        "## Theoretical Maximum Detection",
    ]
    fused = report["metrics"]["fused_anomaly"]
    lines.extend([
        f"Fused ROC AUC: {fused['auc']:.4f}",
        f"Fused overlap coefficient: {fused['overlap_coefficient']:.4f}",
        f"Best balanced accuracy: {fused['best_youden']['balanced_accuracy']:.4f}",
        f"Best F1: {fused['best_f1']['f1']:.4f}",
        f"Best-F1 sensitivity: {fused['best_f1']['sensitivity']:.4f}",
        f"Best-F1 false positive rate: {fused['best_f1']['false_positive_rate']:.4f}",
        "",
        "## Component AUCs",
    ])
    for name, metrics in report["metrics"].items():
        lines.append(f"- {name}: AUC={metrics['auc']:.4f}, overlap={metrics['overlap_coefficient']:.4f}")
    lines.extend([
        "",
        "## Plots",
        *[f"- {Path(path).name}" for path in report["plots"].values()],
    ])
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--models", default="models")
    parser.add_argument("--max-healthy-windows", type=int, default=3000)
    parser.add_argument("--max-labeled-windows-per-file", type=int, default=4000)
    args = parser.parse_args()

    data_dir = Path(args.data)
    model_dir = Path(args.models)
    reports_dir = model_dir / "reports"
    plots_dir = reports_dir / "separation_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    thresholds = json.loads((model_dir / "anomaly_threshold.json").read_text())
    weights = thresholds.get("fusion_weights", {})
    healthy_rows = load_healthy_rows(healthy_dataset_paths(data_dir), args.max_healthy_windows, seed=101)
    labeled_rows = load_labeled_rows(sorted(data_dir.glob("Data_I*.csv")), args.max_labeled_windows_per_file, seed=102)
    rows = healthy_rows + [row for row in labeled_rows if row.label in {"healthy", "anomalous"}]
    scored = score_rows(rows, model_dir, thresholds, weights)
    feature_audit = json.loads((reports_dir / "feature_audit.json").read_text()) if (reports_dir / "feature_audit.json").exists() else {}

    metrics = {}
    plots = {}
    for plot_name, field in COMPONENTS.items():
        healthy = np.asarray([row[field] for row in scored if row["label"] == "healthy"], dtype=float)
        anomalous = np.asarray([row[field] for row in scored if row["label"] == "anomalous"], dtype=float)
        metrics[plot_name] = _max_detection_metrics(healthy, anomalous)
        metrics[plot_name]["healthy_distribution"] = _score_summary([{**row, "score": row[field]} for row in scored if row["label"] == "healthy"]).get("healthy")
        metrics[plot_name]["anomalous_distribution"] = _score_summary([{**row, "score": row[field]} for row in scored if row["label"] == "anomalous"]).get("anomalous")
        threshold = thresholds.get("anomaly_threshold") if plot_name == "fused_anomaly" else None
        plot_path = plots_dir / f"{plot_name}_distribution.png"
        _plot_distribution(plot_name, healthy, anomalous, plot_path, threshold)
        plots[plot_name] = str(plot_path)

    report = {
        "dataset": {
            "healthy_windows": sum(row["label"] == "healthy" for row in scored),
            "anomalous_windows": sum(row["label"] == "anomalous" for row in scored),
            "source": "Healthy MIG Sensor Data plus labeled Data_I*.csv windows.",
        },
        "metrics": metrics,
        "bottleneck": _bottleneck_verdict(metrics, feature_audit, weights),
        "plots": plots,
        "current_thresholds": {
            "watch": thresholds.get("score_bands", {}).get("watch"),
            "warning": thresholds.get("score_bands", {}).get("warning"),
            "critical": thresholds.get("critical_threshold"),
        },
    }
    json_path = reports_dir / "anomaly_score_separation_report.json"
    md_path = reports_dir / "anomaly_score_separation_report.md"
    json_path.write_text(json.dumps(report, indent=2))
    _write_markdown(report, md_path)
    print(json.dumps({
        "report": str(json_path),
        "markdown": str(md_path),
        "plots": plots,
        "bottleneck": report["bottleneck"],
        "fused": metrics["fused_anomaly"],
    }, indent=2))


if __name__ == "__main__":
    main()
