# ============================================================
# FILE: models/neutralad.py   (PY3.8 SAFE)
# ============================================================
# Wraps NeuTraLAD as a "teacher" detector with:
#   - fit(X_normals) -> {"fit_sec": ...}
#   - score(X) -> anomaly scores (higher = more anomalous)

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import DataLoader, Dataset


# -------------------------
# Dataset
# -------------------------
class _TensorDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float()


# -------------------------
# NeuTraLAD core network (your pasted model, minimally cleaned)
# -------------------------
class NeuTraLAD(nn.Module):
    name = "NeuTraLAD"

    def __init__(
        self,
        in_features: int,
        fc_1_out: int = 128,
        fc_last_out: int = 32,
        compression_unit: int = 16,
        n_transforms: int = 4,
        n_layers: int = 3,
        trans_type: str = "mlp",     # 'mlp' in your code; actual behavior uses mask mult or residual
        temperature: float = 0.07,
        trans_fc_in: Optional[int] = None,
        trans_fc_out: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.compression_unit = compression_unit
        self.fc_1_out = fc_1_out
        self.fc_last_out = fc_last_out
        self.n_layers = n_layers
        self.n_transforms = n_transforms
        self.temperature = temperature
        self.trans_type = trans_type

        self.trans_fc_in = trans_fc_in if (trans_fc_in is not None and trans_fc_in > 0) else self.in_features
        self.trans_fc_out = trans_fc_out if (trans_fc_out is not None and trans_fc_out > 0) else self.in_features

        self._build_network()
        self.to(self.device)

    def _create_network(self, D: int, out_dims: list, bias: bool = True) -> list:
        net_layers = []
        prev = D
        for dim in out_dims:
            net_layers.append(nn.Linear(prev, dim, bias=bias))
            net_layers.append(nn.ReLU())
            prev = dim
        return net_layers

    def _create_masks(self) -> list:
        masks = [None] * self.n_transforms
        out_dims = self.trans_layers
        for k in range(self.n_transforms):
            net_layers = self._create_network(self.in_features, out_dims, bias=False)
            net_layers[-1] = nn.Sigmoid()
            masks[k] = nn.Sequential(*net_layers).to(self.device)
        return masks

    def _build_network(self):
        # encoder dims
        out_dims = [0] * self.n_layers
        out_features = self.fc_1_out
        for i in range(self.n_layers - 1):
            out_dims[i] = out_features
            out_features -= self.compression_unit
        out_dims[-1] = self.fc_last_out

        # transform nets dims
        self.trans_layers = [self.trans_fc_in, self.trans_fc_out]

        enc_layers = self._create_network(self.in_features, out_dims)[:-1]  # drop last ReLU
        self.enc = nn.Sequential(*enc_layers).to(self.device)
        self.masks = self._create_masks()

    def _computeX_k(self, X: torch.Tensor) -> torch.Tensor:
        X_t_s = []

        def transform_fn(ttype: str):
            # your code: 'res' => additive, else multiplicative
            if ttype == "res":
                return lambda mask, x: mask(x) + x
            else:
                return lambda mask, x: mask(x) * x

        t_func = transform_fn(self.trans_type)

        for k in range(self.n_transforms):
            X_t_k = t_func(self.masks[k], X)
            X_t_s.append(X_t_k)

        return torch.stack(X_t_s, dim=0)  # [K, B, D]

    def _computeBatchH_ij(self, Zk: torch.Tensor) -> torch.Tensor:
        # Zk: [B, K, H]
        hij = F.cosine_similarity(Zk.unsqueeze(2), Zk.unsqueeze(1), dim=3)  # [B, K, K]
        return torch.exp(hij / self.temperature)

    def _computeBatchH_x_xk(self, z: torch.Tensor, zk: torch.Tensor) -> torch.Tensor:
        # z: [B, H], zk: [B, K, H]
        hij = F.cosine_similarity(z.unsqueeze(1), zk, dim=2)  # [B, K]
        return torch.exp(hij / self.temperature)

    def score(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        Xk = self._computeX_k(X)          # [K, B, D]
        Xk = Xk.permute((1, 0, 2))        # [B, K, D]

        Zk = self.enc(Xk)                 # [B, K, H]
        Zk = F.normalize(Zk, dim=-1)

        Z = self.enc(X)                   # [B, H]
        Z = F.normalize(Z, dim=-1)

        Hij = self._computeBatchH_ij(Zk)          # [B, K, K]
        Hx_xk = self._computeBatchH_x_xk(Z, Zk)   # [B, K]

        mask_not_k = (~torch.eye(self.n_transforms, dtype=torch.bool, device=self.device)).float()  # [K,K]

        numerator = Hx_xk                              # [B,K]
        denominator = Hx_xk + (mask_not_k * Hij).sum(dim=2)  # [B,K]
        scores_V = numerator / (denominator + 1e-12)

        score_V = (-torch.log(scores_V + 1e-12)).sum(dim=1)  # [B]
        return score_V

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.score(X)


# -------------------------
# Config + Detector wrapper
# -------------------------
@dataclass
class NeuTraLADConfig:
    # training
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0

    # model
    fc_1_out: int = 128
    fc_last_out: int = 32
    compression_unit: int = 16
    n_transforms: int = 4
    n_layers: int = 3
    trans_type: str = "mlp"  # if you want residual masks use "res"
    temperature: float = 0.07
    trans_fc_in: Optional[int] = None
    trans_fc_out: Optional[int] = None


class NeuTraLADDetector:
    """
    Provides the same interface your experiment expects:
      - fit(X_normals) -> {"fit_sec": float}
      - score(X) -> np.ndarray scores (higher = more anomalous)
    """

    def __init__(self, d_in: int, cfg: NeuTraLADConfig, device: str = "cuda"):
        self.d_in = int(d_in)
        self.cfg = cfg
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        self.model = NeuTraLAD(
            in_features=self.d_in,
            fc_1_out=cfg.fc_1_out,
            fc_last_out=cfg.fc_last_out,
            compression_unit=cfg.compression_unit,
            n_transforms=cfg.n_transforms,
            n_layers=cfg.n_layers,
            trans_type=cfg.trans_type,
            temperature=cfg.temperature,
            trans_fc_in=cfg.trans_fc_in,
            trans_fc_out=cfg.trans_fc_out,
            device=self.device,
        )

    def fit(self, X: np.ndarray, y=None):
        t0 = time.time()

        X = np.asarray(X, dtype=np.float32)
        ds = _TensorDataset(X)
        loader = DataLoader(ds, batch_size=int(self.cfg.batch_size), shuffle=True, drop_last=False)

        opt = optim.Adam(self.model.parameters(), lr=float(self.cfg.lr), weight_decay=float(self.cfg.weight_decay))

        self.model.train()
        for _epoch in range(int(self.cfg.epochs)):
            for xb in loader:
                xb = xb.to(self.device)
                opt.zero_grad()
                loss = self.model.score(xb).mean()
                loss.backward()
                opt.step()

        t1 = time.time()
        return {"fit_sec": float(t1 - t0)}

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)

        xb = torch.from_numpy(X).to(self.device)
        bs = 65536
        outs = []
        for i in range(0, xb.size(0), bs):
            outs.append(self.model.score(xb[i : i + bs]).detach().cpu().numpy())
        return np.concatenate(outs, axis=0).reshape(-1)
