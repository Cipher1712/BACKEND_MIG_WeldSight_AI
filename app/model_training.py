"""Cross-validated classifier and healthy-weld anomaly model training."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from .features import WindowFeatures
from .vae import PhysicsInformedVAE, physics_vae_loss


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Allow estimators requiring numeric targets to expose original labels."""

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.label_encoder_ = LabelEncoder().fit(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, self.label_encoder_.transform(y))
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.label_encoder_.inverse_transform(self.estimator_.predict(X).astype(int))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.estimator_.feature_importances_


def classifier_candidates(random_state: int = 42) -> dict[str, tuple[Any, dict[str, list]]]:
    candidates: dict[str, tuple[Any, dict[str, list]]] = {
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=random_state),
            {"n_estimators": [150, 250, 400], "max_depth": [None, 8, 16],
             "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.7]},
        ),
        "svm": (
            SVC(probability=True, class_weight="balanced", random_state=random_state),
            {"C": [0.5, 1, 3, 10], "gamma": ["scale", "auto", 0.01, 0.1], "kernel": ["rbf"]},
        ),
        "mlp": (
            MLPClassifier(max_iter=700, early_stopping=False, random_state=random_state),
            {"hidden_layer_sizes": [(48,), (64, 32), (96, 48)], "alpha": [1e-5, 1e-4, 1e-3],
             "learning_rate_init": [3e-4, 1e-3]},
        ),
    }
    try:
        from xgboost import XGBClassifier
        candidates["xgboost"] = (
            LabelEncodedClassifier(XGBClassifier(n_jobs=-1, random_state=random_state, eval_metric="mlogloss")),
            {"estimator__n_estimators": [150, 250, 400], "estimator__max_depth": [3, 5, 7],
             "estimator__learning_rate": [0.03, 0.07, 0.12], "estimator__subsample": [0.75, 1.0]},
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        candidates["lightgbm"] = (
            LGBMClassifier(n_jobs=-1, random_state=random_state, verbosity=-1),
            {"n_estimators": [150, 250, 400], "num_leaves": [15, 31, 63],
             "learning_rate": [0.03, 0.07, 0.12], "subsample": [0.75, 1.0]},
        )
    except ImportError:
        pass
    return candidates


SUPPORTED_LABELS = {
    "stable_arc", "arc_instability", "excessive_spatter", "porosity_risk",
    "heat_input_high", "heat_input_low", "short_circuit_instability",
    "abnormal_arc_behaviour", "unknown_anomaly",
}


def train_best_classifier(
    X: np.ndarray, y: np.ndarray, search_iterations: int = 12, scaler: StandardScaler | None = None
) -> tuple[Any, StandardScaler, dict]:
    unknown = set(np.unique(y)) - SUPPORTED_LABELS
    if unknown:
        raise ValueError(f"unsupported labels: {sorted(unknown)}")
    scaler = scaler or StandardScaler().fit(X)
    scaled = scaler.transform(X)
    min_class = int(np.min(np.unique(y, return_counts=True)[1]))
    if min_class < 2:
        raise ValueError("each defect label needs at least two windows for cross-validation")
    folds = max(2, min(5, min_class))
    cv = StratifiedKFold(folds, shuffle=True, random_state=42)
    results, best_model, best_score = {}, None, -1.0
    for name, (model, parameters) in classifier_candidates().items():
        try:
            search = RandomizedSearchCV(
                model, parameters,
                n_iter=min(search_iterations, int(np.prod([len(v) for v in parameters.values()]))),
                scoring="f1_macro", cv=cv, n_jobs=-1, random_state=42, refit=True,
                error_score="raise",
            )
            search.fit(scaled, y)
            results[name] = {"status": "ok", "f1_macro": float(search.best_score_), "params": search.best_params_}
            if search.best_score_ > best_score:
                best_model, best_score = search.best_estimator_, float(search.best_score_)
        except Exception as exc:
            results[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if best_model is None:
        raise RuntimeError("no classifier could be trained")
    return best_model, scaler, {"selected_model": type(best_model).__name__, "cv_f1_macro": best_score, "candidates": results}


def train_healthy_models(
    X: np.ndarray, violations: np.ndarray, output_dir: str = "models",
    latent_dims: tuple[int, ...] = (4, 6, 8), epochs: int = 80, batch_size: int = 128,
) -> dict:
    if len(X) < 16:
        raise ValueError("at least 16 healthy windows are required for VAE training")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X).astype(np.float32)
    joblib.dump(scaler, out / "scaler.pkl")
    isolation = IsolationForest(n_estimators=250, contamination="auto", random_state=42, n_jobs=-1).fit(scaled)
    joblib.dump(isolation, out / "isolation_forest.pkl")

    split = max(1, int(len(scaled) * 0.85))
    order = np.random.default_rng(42).permutation(len(scaled))
    train_idx, validation_idx = order[:split], order[split:]
    if not len(validation_idx):
        validation_idx = train_idx
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best = None
    for latent_dim in latent_dims:
        torch.manual_seed(42)
        model = PhysicsInformedVAE(scaled.shape[1], latent_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        dataset = TensorDataset(torch.from_numpy(scaled[train_idx]))
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
        feature_mean = torch.as_tensor(scaler.mean_, dtype=torch.float32, device=device)
        feature_scale = torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device)
        model.train()
        for _ in range(epochs):
            for (xb,) in loader:
                xb = xb.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction, mu, log_var = model(xb)
                reconstructed_raw = reconstruction * feature_scale + feature_mean
                physics_penalty = torch.stack((
                    torch.relu((70.0 - reconstructed_raw[:, 14]) / 70.0),
                    torch.relu((reconstructed_raw[:, 6] - 1.5) / 3.0),
                    torch.relu((reconstructed_raw[:, 16] - 0.18) / 0.45),
                    torch.relu((reconstructed_raw[:, 25] - 0.72) / 0.28),
                ), dim=1)
                loss, _ = physics_vae_loss(reconstruction, xb, mu, log_var, physics_penalty)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation = torch.from_numpy(scaled[validation_idx]).to(device)
            reconstruction, mu, _ = model(validation)
            validation_loss = float(torch.mean((reconstruction - validation) ** 2).cpu())
        if best is None or validation_loss < best[0]:
            best = (validation_loss, latent_dim, model)

    validation_loss, latent_dim, model = best
    with torch.inference_mode():
        all_x = torch.from_numpy(scaled).to(device)
        reconstruction, latent, _ = model(all_x)
        rec_errors = torch.mean((reconstruction - all_x) ** 2, dim=1).cpu().numpy()
        latent_np = latent.cpu().numpy()
    latent_center, latent_scale = latent_np.mean(0), latent_np.std(0) + 1e-6
    latent_distances = np.sqrt(np.mean(((latent_np - latent_center) / latent_scale) ** 2, axis=1))
    torch.save({"state_dict": model.cpu().state_dict(), "input_dim": scaled.shape[1],
                "latent_dim": latent_dim, "hidden_dim": 48, "latent_center": latent_center,
                "latent_scale": latent_scale, "feature_names": WindowFeatures.names()}, out / "vae.pt")
    thresholds = {
        "reconstruction_threshold": float(np.quantile(rec_errors, 0.995)),
        "latent_threshold": float(np.quantile(latent_distances, 0.995)),
        "anomaly_threshold": 0.60, "ewma_k": 3.0, "ewma_floor": 0.1,
    }
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    return {"latent_dim": latent_dim, "validation_reconstruction": validation_loss, **thresholds}
