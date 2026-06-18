"""Train Isolation Forest and physics-informed VAE on healthy voltage windows."""
import argparse
import json

from app.datasets import load_feature_dataset
from app.model_training import train_healthy_models

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Healthy CSV/Parquet/JSON dataset")
parser.add_argument("--output", default="models")
parser.add_argument("--epochs", type=int, default=80)
args = parser.parse_args()
X, _, violations = load_feature_dataset(args.dataset, require_labels=False)
print(json.dumps(train_healthy_models(X, violations, args.output, epochs=args.epochs), indent=2))
