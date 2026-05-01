# ============================================================
# FILE: attacks/tabextractor.py
# ============================================================


from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Config
# -----------------------------
@dataclass
class TabExtractorConfig:
    # training
    iterations: int = 1000
    batch_size: int = 128
    seed: int = 42

    # generator latent
    z_dim: int = 64

    # paper-style pseudo labeling
    label_threshold_mode: str = "median"   # supports "median"

    # bounds
    use_feat_bounds: bool = True

    # learning rates
    lr_clone: float = 1e-3
    lr_gen: float = 1e-3

    # update schedule
    clone_steps: int = 1                  # number of clone updates per iteration
    gen_steps: int = 1                    # number of generator updates per iteration

    # regularizers to prevent collapse
    balance_weight: float = 0.2           # encourage ~50/50 pseudo labels in gen batch
    entropy_weight: float = 0.0           # optional: encourage uncertain clone preds on gen samples

    # misc
    grad_clip: float = 1.0
    log_every: int = 50

    # milestones (trajectory logging)
    milestones: Optional[List[int]] = None
    milestone_callback: Optional[Callable[[Dict[str, Any]], None]] = None


# -----------------------------
# (Your existing model classes)
# -----------------------------
class TabularGeneratorNum(nn.Module):
    def __init__(self, d_in: int, z_dim: int = 64, hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        layers: List[nn.Module] = []
        prev = int(z_dim)
        for h in hidden:
            layers += [nn.Linear(prev, int(h)), nn.ReLU()]
            prev = int(h)
        layers += [nn.Linear(prev, int(d_in))]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CTTClone(nn.Module):
    """
    Keep YOUR original implementation in your repo.
    This placeholder only matches the interface.
    """
    def __init__(
        self,
        d_in: int,
        n_classes: int = 2,
        d_embed: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(d_in), 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, int(n_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Helpers
# -----------------------------
def _clamp_bounds(x: torch.Tensor, feat_lo: torch.Tensor, feat_hi: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(x, feat_hi), feat_lo)


def _build_stats_snapshot(total_budget: int, used: int, t0: float) -> Dict[str, Any]:
    sec = float(time.time() - t0)
    qps = float(used / max(sec, 1e-9))
    ms_per_q = float(1000.0 * sec / max(used, 1))
    return {
        "query_budget_expected": int(total_budget),
        "query_budget": int(used),
        "extract_sec": float(sec),
        "queries_per_sec": float(qps),
        "ms_per_query": float(ms_per_q),
    }


def _pseudo_labels_from_teacher_scores(y_t: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Convert teacher scores to pseudo binary labels using median thresholding per batch.
    y_t: (bs,)
    returns: (bs,) float in {0,1}
    """
    y_t = y_t.detach().reshape(-1)
    if y_t.numel() == 0:
        return torch.zeros_like(y_t)

    mode = (mode or "median").strip().lower()
    if mode == "median":
        thr = torch.median(y_t)
    else:
        thr = torch.median(y_t)

    return (y_t >= thr).float()


def _clone_logit1(clone_logits: torch.Tensor) -> torch.Tensor:
    """
    Support either (N,2) logits or (N,) / (N,1).
    Return a 1D logit used for BCE-with-logits.
    """
    if clone_logits.ndim == 2 and clone_logits.size(1) == 2:
        return clone_logits[:, 1].reshape(-1)
    return clone_logits.reshape(-1)


def _entropy_from_probs(p: torch.Tensor) -> torch.Tensor:
    # p in (0,1)
    return -(p * torch.log(p + 1e-8) + (1.0 - p) * torch.log(1.0 - p + 1e-8)).mean()


# -----------------------------
# Training loop (paper-faithful)
# -----------------------------
def train_student_tabextractor(
    teacher_score_fn,                 # x -> score (N,)
    clone: nn.Module,
    generator: nn.Module,
    device: torch.device,
    cfg: TabExtractorConfig,
    feat_lo: torch.Tensor,
    feat_hi: torch.Tensor,
) -> Dict[str, Any]:
    """
    Paper-faithful alternating optimisation:
      - Clone step: generate x_c -> query teacher -> pseudo labels -> train clone (BCE)
      - Gen step  : generate x_g -> query teacher -> pseudo labels -> update generator
                    to maximize clone loss (hard samples) + anti-collapse regularizers

    Query accounting:
      used += batch_size for every teacher call
      total_budget = iterations * batch_size * (clone_steps + gen_steps)
    """
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    clone = clone.to(device).train()
    generator = generator.to(device).train()

    opt_c = torch.optim.Adam(clone.parameters(), lr=float(cfg.lr_clone))
    opt_g = torch.optim.Adam(generator.parameters(), lr=float(cfg.lr_gen))

    bs = int(cfg.batch_size)
    iters = int(cfg.iterations)
    z_dim = int(cfg.z_dim)

    # faithful query budget accounting
    total_budget = int(iters * bs * (max(0, int(cfg.clone_steps)) + max(0, int(cfg.gen_steps))))

    milestones = None
    if cfg.milestones is not None:
        milestones = sorted([int(m) for m in cfg.milestones if int(m) > 0])

    used = 0
    t0 = time.time()

    last_loss_c = torch.tensor(float("nan"), device=device)
    last_loss_g = torch.tensor(float("nan"), device=device)

    for it in range(1, iters + 1):

        # =====================================================
        # (A) CLONE STEP(S): teacher-supervised pseudo labels
        # =====================================================
        for _ in range(int(cfg.clone_steps)):
            z = torch.randn(bs, z_dim, device=device)
            x = generator(z)
            if bool(cfg.use_feat_bounds):
                x = _clamp_bounds(x, feat_lo, feat_hi)

            with torch.no_grad():
                y_t = teacher_score_fn(x).reshape(-1)

            used += int(x.size(0))

            y_bin = _pseudo_labels_from_teacher_scores(y_t, cfg.label_threshold_mode)

            clone.train()
            logits = clone(x)
            logit1 = _clone_logit1(logits)

            loss_c = F.binary_cross_entropy_with_logits(logit1, y_bin)

            opt_c.zero_grad(set_to_none=True)
            loss_c.backward()
            if float(cfg.grad_clip) and float(cfg.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(clone.parameters(), float(cfg.grad_clip))
            opt_c.step()

            last_loss_c = loss_c.detach()

        # =====================================================
        # (B) GENERATOR STEP(S): adversarial + anti-collapse
        # =====================================================
        for _ in range(int(cfg.gen_steps)):
            z = torch.randn(bs, z_dim, device=device)
            xg = generator(z)
            if bool(cfg.use_feat_bounds):
                xg = _clamp_bounds(xg, feat_lo, feat_hi)

            # **Faithful**: teacher queried for generator batch too
            with torch.no_grad():
                y_tg = teacher_score_fn(xg).reshape(-1)

            used += int(xg.size(0))

            yb_g = _pseudo_labels_from_teacher_scores(y_tg, cfg.label_threshold_mode)

            # Freeze clone weights, but allow gradient through clone wrt xg
            clone.eval()
            logits_g = clone(xg)
            logit1_g = _clone_logit1(logits_g)

            # Generator wants to create hard samples:
            # maximize clone loss => minimize (-loss)
            ce = F.binary_cross_entropy_with_logits(logit1_g, yb_g)

            # balance regularizer: encourage pseudo labels not to collapse to all 0 or all 1
            # target ~0.5 positives
            pos_rate = yb_g.mean()
            balance = torch.abs(pos_rate - 0.5)

            # optional entropy term on clone probs (encourage uncertainty near boundary)
            p = torch.sigmoid(logit1_g)
            ent = _entropy_from_probs(p)

            # We MINIMIZE loss_g, so use negative CE to maximize CE
            loss_g = (-ce) + float(cfg.balance_weight) * balance + float(cfg.entropy_weight) * (-ent)

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            if float(cfg.grad_clip) and float(cfg.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), float(cfg.grad_clip))
            opt_g.step()

            last_loss_g = loss_g.detach()

        # =====================================================
        # logging
        # =====================================================
        if int(cfg.log_every) > 0 and (it % int(cfg.log_every) == 0):
            print(
                f"[TabExtractor] iter={it}/{iters} "
                f"used={used}/{total_budget} "
                f"loss_c={float(last_loss_c.item()):.6f} "
                f"loss_g={float(last_loss_g.item()):.6f}"
            )

        # =====================================================
        # milestone callback (trajectory logging)
        # =====================================================
        if milestones and cfg.milestone_callback:
            while milestones and used >= int(milestones[0]):
                snap = _build_stats_snapshot(total_budget=total_budget, used=used, t0=t0)
                cfg.milestone_callback({"used": int(used), "stats": snap})
                milestones.pop(0)

        # stop if we exactly hit the expected budget (optional safety)
        if used >= total_budget:
            break

    stats = _build_stats_snapshot(total_budget=total_budget, used=used, t0=t0)
    return stats


__all__ = [
    "TabExtractorConfig",
    "TabularGeneratorNum",
    "CTTClone",
    "train_student_tabextractor",
]
