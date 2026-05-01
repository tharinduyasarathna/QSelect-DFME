
# ============================================================
# FILE: attacks/query_selectors/query_selector_gpu_shared_projection.py
# ============================================================

import torch
from dataclasses import dataclass


@dataclass
class GPUSharedProjCfg:
    dim: int = 8
    seed: int = 0
    prefilter_ratio: float = 0.25
    candidate_factor: int = 4
    use_cosine: bool = True
    eps: float = 1e-8


class GPUSharedProjectionSelector:
    def __init__(self, d_in: int, cfg: GPUSharedProjCfg, device: torch.device):
        self.cfg = cfg
        self.device = device

        dim = int(min(max(1, cfg.dim), d_in))
        g = torch.Generator(device=device)
        g.manual_seed(int(cfg.seed))

        W = torch.randn(d_in, dim, generator=g, device=device)
        W = W / (W.norm(dim=0, keepdim=True) + cfg.eps)
        self.W = W

        self._sel_gen = torch.Generator(device=device)
        self._sel_gen.manual_seed(int(cfg.seed) + 12345)

    @torch.no_grad()
    def _project(self, X: torch.Tensor) -> torch.Tensor:
        Z = X @ self.W
        if self.cfg.use_cosine:
            Z = Z / (Z.norm(dim=1, keepdim=True) + self.cfg.eps)
        return Z

    @torch.no_grad()
    def _uncertainty(self, s: torch.Tensor) -> torch.Tensor:
        mu = s.mean()
        sd = s.std().clamp_min(1e-6)
        z = (s - mu) / sd
        return torch.exp(-z.abs())

    @torch.no_grad()
    def _farthest_first_cosine(self, Z: torch.Tensor, k: int) -> torch.Tensor:
        N = Z.shape[0]
        if k >= N:
            return torch.arange(N, device=Z.device)

        start = int(torch.randint(0, N, (1,), generator=self._sel_gen, device=Z.device).item())
        sel = torch.empty((k,), dtype=torch.long, device=Z.device)
        sel[0] = start

        sim = (Z @ Z[start].unsqueeze(1)).squeeze(1)
        d = 1.0 - sim

        for i in range(1, k):
            j = int(torch.argmax(d).item())
            sel[i] = j
            sim_j = (Z @ Z[j].unsqueeze(1)).squeeze(1)
            d = torch.minimum(d, 1.0 - sim_j)
        return sel

    @torch.no_grad()
    def select(self, X_pool: torch.Tensor, k: int, student_scores: torch.Tensor) -> torch.Tensor:
        N = int(X_pool.shape[0])
        k = int(min(max(k, 0), N))
        if k <= 0:
            return torch.empty((0,), dtype=torch.long, device=X_pool.device)

        u = self._uncertainty(student_scores)

        m1 = int(max(k * int(self.cfg.candidate_factor), 1))
        m2 = int(round(N * float(self.cfg.prefilter_ratio)))
        m = int(min(max(m1, m2), N))

        top_idx = torch.topk(u, k=m, largest=True).indices
        Z = self._project(X_pool.index_select(0, top_idx))
        pick_local = self._farthest_first_cosine(Z, k=k)
        return top_idx[pick_local]

