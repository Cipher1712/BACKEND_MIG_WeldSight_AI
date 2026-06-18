"""Evaluate trained anomaly artifacts on real voltage windows only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.inference import InferencePipeline
from app.welding_data import condition_dataset_paths, healthy_dataset_paths, iter_real_windows


def summarize_windows(pipeline: InferencePipeline, paths: list[Path], max_windows_per_file: int) -> dict:
    rows = []
    for path in paths:
        scores, qualities, statuses = [], [], {}
        for index, window in enumerate(iter_real_windows(path)):
            if index >= max_windows_per_file:
                break
            result = pipeline.predict(window.tolist())
            scores.append(result["anomaly_score"])
            qualities.append(result["quality_score"])
            statuses[result["status"]] = statuses.get(result["status"], 0) + 1
        if scores:
            rows.append({
                "file": str(path), "windows": len(scores),
                "anomaly_score_mean": float(np.mean(scores)),
                "anomaly_score_p95": float(np.quantile(scores, 0.95)),
                "quality_score_mean": float(np.mean(qualities)),
                "statuses": statuses,
            })
    total_windows = sum(row["windows"] for row in rows)
    return {"files": rows, "total_windows": total_windows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--models", default="models")
    parser.add_argument("--max-windows-per-file", type=int, default=500)
    args = parser.parse_args()

    pipeline = InferencePipeline(args.models)
    if not pipeline.ready:
        raise SystemExit(f"Model artifacts are incomplete: {pipeline.health()}")
    report = {
        "model_health": pipeline.health(),
        "healthy": summarize_windows(pipeline, healthy_dataset_paths(args.data), args.max_windows_per_file),
        "conditions": summarize_windows(pipeline, condition_dataset_paths(args.data), args.max_windows_per_file),
    }
    output = Path(args.models) / "evaluation_report.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
