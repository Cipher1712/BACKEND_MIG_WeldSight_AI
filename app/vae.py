"""Compact physics-informed variational autoencoder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PhysicsInformedVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 6, hidden_dim: int = 48):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.log_var = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
                                     nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, input_dim))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var) if self.training else mu
        return self.decoder(z), mu, log_var


def physics_vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    physics_violations: torch.Tensor | None = None,
    beta: float = 0.01,
    physics_weight: float = 0.15,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction_loss = F.mse_loss(reconstruction, target)
    kl_loss = -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())
    physics_loss = torch.mean(torch.relu(physics_violations)) if physics_violations is not None else target.new_tensor(0.0)
    total = reconstruction_loss + beta * kl_loss + physics_weight * physics_loss
    return total, {"reconstruction": float(reconstruction_loss.detach()),
                   "kl": float(kl_loss.detach()), "physics": float(physics_loss.detach())}


@dataclass
class VAEInference:
    model: PhysicsInformedVAE
    latent_center: np.ndarray
    latent_scale: np.ndarray
    device: str = "cpu"

    def anomaly_components(self, features: np.ndarray) -> tuple[float, float]:
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.inference_mode():
            reconstruction, mu, _ = self.model(x)
        rec = float(torch.mean((reconstruction - x) ** 2, dim=1)[0].cpu())
        z = mu.cpu().numpy()[0]
        latent = float(np.sqrt(np.mean(((z - self.latent_center) / (self.latent_scale + 1e-6)) ** 2)))
        return rec, latent


def load_vae(path: str, device: str = "cpu") -> VAEInference:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = PhysicsInformedVAE(payload["input_dim"], payload["latent_dim"], payload.get("hidden_dim", 48))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return VAEInference(model, np.asarray(payload["latent_center"]), np.asarray(payload["latent_scale"]), device)
