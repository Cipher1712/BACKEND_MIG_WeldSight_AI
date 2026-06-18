"""Evaluate exported classifier artifacts."""
import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize

from app.datasets import load_feature_dataset

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--models", default="models")
args = parser.parse_args()
X, y, _ = load_feature_dataset(args.dataset)
root = Path(args.models)
model, scaler = joblib.load(root / "classifier.pkl"), joblib.load(root / "scaler.pkl")
scaled = scaler.transform(X)
prediction, probability = model.predict(scaled), model.predict_proba(scaled)
precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average="macro", zero_division=0)
classes = model.classes_
try:
    auc = roc_auc_score(label_binarize(y, classes=classes), probability, average="macro", multi_class="ovr")
except ValueError:
    auc = None
report = {"accuracy": accuracy_score(y, prediction), "precision_macro": precision,
          "recall_macro": recall, "f1_macro": f1, "roc_auc_ovr_macro": auc,
          "classes": classes.tolist(), "confusion_matrix": confusion_matrix(y, prediction, labels=classes).tolist()}
(root / "evaluation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
