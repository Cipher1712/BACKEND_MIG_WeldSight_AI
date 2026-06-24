"""Train WeldSight anomaly artifacts from real voltage-only welding datasets."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model_training import train_healthy_models
from app.welding_data import healthy_dataset_paths, load_real_feature_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="Directory containing MIG Sensor Data/")
    parser.add_argument("--output", default="models")
    parser.add_argument("--material", default=None)
    parser.add_argument("--thickness-mm", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--max-windows", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    paths = healthy_dataset_paths(args.data)
    if not paths:
        raise SystemExit(f"No healthy CSV files found in {Path(args.data) / 'MIG Sensor Data'}")
    X, summaries = load_real_feature_matrix(paths, max_windows=args.max_windows)
    output_dir = args.output
    if args.material and args.thickness_mm is not None:
        key = "".join(ch.lower() if ch.isalnum() else "_" for ch in args.material).strip("_")
        key = f"{key}_{args.thickness_mm:.2f}mm".replace(".", "p")
        output_dir = str(Path(args.output) / "profiles" / key)
    report = train_healthy_models(
        X, output_dir, material=args.material, thickness_mm=args.thickness_mm,
        epochs=args.epochs, batch_size=args.batch_size,
    )
    report["source"] = "real_voltage_only"
    report["datasets"] = [asdict(summary) for summary in summaries]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
