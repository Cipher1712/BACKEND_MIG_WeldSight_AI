"""Low-overhead local explanations, with SHAP when available."""
from __future__ import annotations

from typing import Any

import numpy as np


class Explainer:
    def __init__(self, model: Any, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self._explainer = None
        self._kind = "fallback"
        try:
            import shap
            tree_model = getattr(model, "estimator_", model)
            try:
                self._explainer = shap.TreeExplainer(tree_model)
                self._kind = "tree"
            except Exception:
                background = np.zeros((1, len(feature_names)), dtype=float)
                self._explainer = shap.Explainer(model.predict_proba, background, algorithm="permutation")
                self._kind = "permutation"
        except Exception:
            self._explainer = None

    def explain(self, x: np.ndarray, prediction: str, top_k: int = 3) -> list[dict[str, float | str]]:
        row = np.asarray(x, dtype=float).reshape(1, -1)
        values: np.ndarray
        if self._explainer is not None:
            try:
                class_idx = list(self.model.classes_).index(prediction)
                if self._kind == "tree":
                    raw = self._explainer.shap_values(row)
                    if isinstance(raw, list):
                        values = np.asarray(raw[class_idx])[0]
                    else:
                        arr = np.asarray(raw)
                        values = arr[0, :, class_idx] if arr.ndim == 3 else arr[0]
                else:
                    raw = self._explainer(row, max_evals=2 * row.shape[1] + 1)
                    arr = np.asarray(raw.values)
                    values = arr[0, :, class_idx] if arr.ndim == 3 else arr[0]
            except Exception:
                values = self._fallback_values(row, prediction)
        else:
            values = self._fallback_values(row, prediction)
        indices = np.argsort(np.abs(values))[-top_k:][::-1]
        return [{"feature": self.feature_names[i], "impact": round(float(values[i]), 5),
                 "value": round(float(row[0, i]), 5)} for i in indices]

    def _fallback_values(self, row: np.ndarray, prediction: str) -> np.ndarray:
        if hasattr(self.model, "feature_importances_"):
            values = np.asarray(self.model.feature_importances_) * np.abs(row[0])
        elif hasattr(self.model, "coef_"):
            class_idx = list(self.model.classes_).index(prediction)
            values = np.asarray(self.model.coef_[class_idx]) * row[0]
        else:
            values = np.abs(row[0])
        return values


def explanation_text(prediction: str, top_features: list[dict]) -> str:
    friendly = {
        "spectral_entropy": "spectral entropy", "short_circuit_ratio": "short-circuit ratio",
        "arc_stability_index": "arc stability", "std_v": "voltage variation",
        "spike_density": "voltage spike density", "mean_v": "mean voltage",
    }
    names = [friendly.get(str(item["feature"]), str(item["feature"]).replace("_", " ")) for item in top_features[:2]]
    evidence = " and ".join(names) if names else "the measured voltage pattern"
    return f"{evidence.capitalize()} most strongly support {prediction.replace('_', ' ')}."
