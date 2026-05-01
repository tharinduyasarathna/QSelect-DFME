# models/drocc.py
# DROCC teacher for tabular one-class anomaly detection
# API: fit(X_norm)->dict, score(X)->np.ndarray (higher = more anomalous)

import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


@dataclass
class DROCCConfig:
    hid_dim: int = 128
    epochs: int = 10
    batch_size: int = 1024
    lr: float = 1e-3

    # DROCC-specific
    lamda: float = 1.0
    radius: float = 0.2
    gamma: float = 2.0
    ascent_step_size: float = 1e-3
    ascent_num_steps: int = 50


class _MLP(nn.Module):
    def __init__(self, d_in: int, hid_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)  # logits (B,)


class DROCCDetector:
    """
    One-class DROCC-style training:
      - Train on normals only (targets=0)
      - Generate adversarial samples around normals and penalize them (also towards 0) as in the provided code variant.
    Score:
      - returns logits (higher => more anomalous, typically)
    """
    def __init__(self, d_in: int, cfg: DROCCConfig, device: str = "cuda"):
        self.d_in = int(d_in)
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        self.model = _MLP(self.d_in, cfg.hid_dim).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)

    def _one_class_adv_loss(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Adversarial ascent around normals, with projection to [radius, gamma*radius].
        """
        B = x_norm.size(0)
        # Start from random noise around x_norm
        x_adv = torch.randn_like(x_norm, device=self.device)
        x_adv_sampled = (x_adv + x_norm).detach().requires_grad_(True)

        for step in range(self.cfg.ascent_num_steps):
            logits = self.model(x_adv_sampled)
            targets = torch.zeros(B, device=self.device)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

            grad = torch.autograd.grad(loss, [x_adv_sampled], retain_graph=False, create_graph=False)[0]
            grad_norm = torch.norm(grad, p=2, dim=1, keepdim=True)  # (B,1)
            grad_normalized = grad / (grad_norm + 1e-8)

            with torch.no_grad():
                x_adv_sampled.add_(self.cfg.ascent_step_size * grad_normalized)

                # project perturbation
                h = x_adv_sampled - x_norm
                norm_h = torch.norm(h, p=2, dim=1, keepdim=True)  # (B,1)
                alpha = torch.clamp(norm_h, self.cfg.radius, self.cfg.gamma * self.cfg.radius)
                proj = alpha / (norm_h + 1e-8)
                h = proj * h
                x_adv_sampled = (x_norm + h).detach().requires_grad_(True)

        adv_logits = self.model(x_adv_sampled)
        adv_loss = F.binary_cross_entropy_with_logits(adv_logits, torch.zeros_like(adv_logits))
        return adv_loss

    def fit(self, X: np.ndarray):
        t0 = time.time()

        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        idx = np.arange(n)

        self.model.train()
        for epoch in range(int(self.cfg.epochs)):
            np.random.shuffle(idx)
            for i in range(0, n, int(self.cfg.batch_size)):
                batch = X[idx[i:i + int(self.cfg.batch_size)]]
                x = torch.from_numpy(batch).to(self.device)

                # normals-only target = 0
                logits = self.model(x)
                y0 = torch.zeros_like(logits)
                ce_loss = F.binary_cross_entropy_with_logits(logits, y0)

                adv_loss = self._one_class_adv_loss(x)
                loss = ce_loss + self.cfg.lamda * adv_loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        t1 = time.time()
        return {"fit_sec": float(t1 - t0)}

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly score; higher should mean more anomalous.
        """
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(X).to(self.device)
            logits = self.model(x).detach().cpu().numpy().reshape(-1)
        return logits
