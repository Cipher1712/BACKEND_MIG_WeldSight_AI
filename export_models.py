"""Validate WeldSight anomaly artifact bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.inference import InferencePipeline

parser = argparse.ArgumentParser()
parser.add_argument("--models", default="models")
args = parser.parse_args()
root = Path(args.models)
required = ["vae.pt", "scaler.pkl", "anomaly_threshold.json", "isolation_forest.pkl"]
missing = [name for name in required if not (root / name).exists()]
if missing:
    raise SystemExit(f"Missing artifacts: {', '.join(missing)}")
pipeline = InferencePipeline(root)
manifest = {"version": 1, **pipeline.health(), "artifacts": {}}
for name in required:
    data = (root / name).read_bytes()
    manifest["artifacts"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
