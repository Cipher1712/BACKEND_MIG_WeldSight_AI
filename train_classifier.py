"""Train and select the best supported defect classifier."""
import argparse
import json
from pathlib import Path

import joblib

from app.datasets import load_feature_dataset
from app.model_training import train_best_classifier

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Labeled CSV/Parquet/JSON dataset")
parser.add_argument("--output", default="models")
parser.add_argument("--search-iterations", type=int, default=12)
args = parser.parse_args()
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
X, y, _ = load_feature_dataset(args.dataset)
existing_scaler = joblib.load(out / "scaler.pkl") if (out / "scaler.pkl").exists() else None
model, scaler, report = train_best_classifier(X, y, args.search_iterations, existing_scaler)
joblib.dump(model, out / "classifier.pkl")
joblib.dump(scaler, out / "scaler.pkl")
(out / "classifier_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
