"""Healthy-only Isolation Forest and physics-informed VAE training."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .features import WindowFeatures
from .vae import PhysicsInformedVAE, physics_vae_loss


def _physics_penalty(reconstructed_raw: torch.Tensor) -> torch.Tensor:
    """Soft constraints tied to voltage-only arc behavior."""
    names = WindowFeatures.names()
    idx = {name: names.index(name) for name in names}
    return torch.stack((
        torch.relu((55.0 - reconstructed_raw[:, idx["arc_stability_index"]]) / 55.0),
        torch.relu((reconstructed_raw[:, idx["std_v"]] - 8.0) / 8.0),
        torch.relu((reconstructed_raw[:, idx["short_circuit_ratio"]] - 0.35) / 0.35),
        torch.relu((reconstructed_raw[:, idx["spectral_entropy"]] - 0.90) / 0.10),
    ), dim=1)


def train_healthy_models(
    X: np.ndarray,
    output_dir: str = "models",
    material: str | None = None,
    thickness_mm: float | None = None,
    latent_dims: tuple[int, ...] = (4, 6, 8),
    epochs: int = 25,
    batch_size: int = 256,
) -> dict:
    if len(X) < 100:
        raise ValueError("at least 100 real healthy windows are required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    order = rng.permutation(len(X))
    validation_size = max(100, int(0.15 * len(X)))
    validation_idx, train_idx = order[:validation_size], order[validation_size:]

    scaler = StandardScaler().fit(X[train_idx])
    scaled = scaler.transform(X).astype(np.float32)
    joblib.dump(scaler, output / "scaler.pkl")

    isolation = IsolationForest(
        n_estimators=300, max_samples="auto", contamination="auto",
        random_state=42, n_jobs=1,
    ).fit(scaled[train_idx])
    joblib.dump(isolation, output / "isolation_forest.pkl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tensor = torch.from_numpy(scaled[train_idx])
    loader = DataLoader(
        TensorDataset(train_tensor), batch_size=min(batch_size, len(train_tensor)),
        shuffle=True, generator=torch.Generator().manual_seed(42),
    )
    feature_mean = torch.as_tensor(scaler.mean_, dtype=torch.float32, device=device)
    feature_scale = torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device)
    best: tuple[float, int, PhysicsInformedVAE] | None = None

    for latent_dim in latent_dims:
        torch.manual_seed(42)
        model = PhysicsInformedVAE(scaled.shape[1], latent_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
        for _ in range(epochs):
            model.train()
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction, mu, log_var = model(batch)
                raw_reconstruction = reconstruction * feature_scale + feature_mean
                loss, _ = physics_vae_loss(
                    reconstruction, batch, mu, log_var,
                    _physics_penalty(raw_reconstruction), beta=0.01, physics_weight=0.08,
                )
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation = torch.from_numpy(scaled[validation_idx]).to(device)
            reconstruction, _, _ = model(validation)
            validation_loss = float(torch.mean((reconstruction - validation) ** 2).cpu())
        if best is None or validation_loss < best[0]:
            best = validation_loss, latent_dim, model

    assert best is not None
    validation_loss, latent_dim, model = best
    model.eval()
    with torch.inference_mode():
        all_tensor = torch.from_numpy(scaled).to(device)
        reconstruction, latent, _ = model(all_tensor)
        reconstruction_errors = torch.mean((reconstruction - all_tensor) ** 2, dim=1).cpu().numpy()
        latent_values = latent.cpu().numpy()
    latent_center = latent_values[train_idx].mean(axis=0)
    latent_scale = latent_values[train_idx].std(axis=0) + 1e-6
    latent_distances = np.sqrt(np.mean(((latent_values - latent_center) / latent_scale) ** 2, axis=1))
    isolation_raw = -isolation.decision_function(scaled)
    feature_names = WindowFeatures.names()
    validation_scaled = scaled[validation_idx].copy()
    base_forest = -isolation.decision_function(validation_scaled)
    rng_importance = np.random.default_rng(43)
    forest_importance = []
    for index in range(validation_scaled.shape[1]):
        shuffled = validation_scaled.copy()
        rng_importance.shuffle(shuffled[:, index])
        shifted = -isolation.decision_function(shuffled)
        forest_importance.append(float(np.mean(np.abs(shifted - base_forest))))
    vae_importance = np.mean((reconstruction.cpu().numpy() - scaled) ** 2, axis=0)
    combined_importance = np.asarray(forest_importance) + np.asarray(vae_importance)
    if float(combined_importance.sum()) > 0.0:
        combined_importance = combined_importance / combined_importance.sum()

    torch.save({
        "state_dict": model.cpu().state_dict(), "input_dim": scaled.shape[1],
        "latent_dim": latent_dim, "hidden_dim": 48, "latent_center": latent_center,
        "latent_scale": latent_scale, "feature_names": feature_names,
    }, output / "vae.pt")

    reference = {}
    for index, name in enumerate(feature_names):
        reference[name] = {
            "q01": float(np.quantile(X[:, index], 0.01)),
            "q50": float(np.quantile(X[:, index], 0.50)),
            "q99": float(np.quantile(X[:, index], 0.99)),
        }
    thresholds = {
        "version": 2,
        "feature_count": int(X.shape[1]),
        "feature_names": feature_names,
        "material": material,
        "thickness_mm": thickness_mm,
        "training_windows": int(len(X)),
        "latent_dim": int(latent_dim),
        "reconstruction": {
            "median": float(np.median(reconstruction_errors[train_idx])),
            "q95": float(np.quantile(reconstruction_errors[train_idx], 0.95)),
            "q995": float(np.quantile(reconstruction_errors[train_idx], 0.995)),
        },
        "latent_distance": {
            "median": float(np.median(latent_distances[train_idx])),
            "q95": float(np.quantile(latent_distances[train_idx], 0.95)),
            "q995": float(np.quantile(latent_distances[train_idx], 0.995)),
        },
        "isolation_forest": {
            "median": float(np.median(isolation_raw[train_idx])),
            "q95": float(np.quantile(isolation_raw[train_idx], 0.95)),
            "q995": float(np.quantile(isolation_raw[train_idx], 0.995)),
        },
        "anomaly_threshold": 0.60,
        "score_bands": {
            "normal": 0.45,
            "watch": 0.60,
            "warning": 0.78,
        },
        "fusion_weights": {
            "vae": 0.42,
            "isolation_forest": 0.24,
            "ewma": 0.18,
            "physics": 0.16,
        },
        "ewma_k": 3.0,
        "feature_reference": reference,
        "feature_importance": [
            {
                "feature": name,
                "importance": float(combined_importance[index]),
                "forest_permutation": float(forest_importance[index]),
                "vae_reconstruction": float(vae_importance[index]),
            }
            for index, name in sorted(
                enumerate(feature_names), key=lambda item: combined_importance[item[0]], reverse=True
            )
        ],
    }
    (output / "anomaly_threshold.json").write_text(json.dumps(thresholds, indent=2))
    return {
        "training_windows": len(X), "validation_windows": len(validation_idx),
        "latent_dim": latent_dim, "validation_reconstruction": validation_loss,
        "device": device, "artifacts": [
            "vae.pt", "scaler.pkl", "isolation_forest.pkl", "anomaly_threshold.json",
        ],
    }
