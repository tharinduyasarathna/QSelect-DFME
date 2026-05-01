# ============================================================
# FILE: attacks/tempest.py
# ============================================================


from dataclasses import dataclass
from typing import Callable, Dict
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TempestConfig:
    query_budget: int = 10000
    gen_mode: str = "gen_var"       # "gen_var" | "gen_min"
    adv_norm: str = "standard"      # "none" | "standard" | "minmax"
    epochs: int = 20
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    var_eps: float = 1e-6
    clip_std: float = 6.0


@dataclass
class PublicStats:
    mean: np.ndarray
    var: np.ndarray
    minv: np.ndarray
    maxv: np.ndarray


def compute_public_stats(X_public: np.ndarray) -> PublicStats:
    X = np.asarray(X_public, dtype=np.float32)
    return PublicStats(
        mean=np.mean(X, axis=0),
        var=np.var(X, axis=0),
        minv=np.min(X, axis=0),
        maxv=np.max(X, axis=0),
    )


def normalize_adv(X: np.ndarray, stats: PublicStats, mode: str) -> np.ndarray:
    mode = (mode or "none").lower().strip()
    X = np.asarray(X, dtype=np.float32)

    if mode == "none":
        return X

    if mode == "standard":
        std = np.sqrt(np.maximum(stats.var, 1e-12)).astype(np.float32)
        return (X - stats.mean.astype(np.float32)) / std

    if mode == "minmax":
        denom = (stats.maxv - stats.minv).astype(np.float32)
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        return (X - stats.minv.astype(np.float32)) / denom

    raise ValueError(f"Unknown adv_norm={mode}")


def generate_queries_from_stats(
    n: int,
    stats: PublicStats,
    gen_mode: str,
    seed: int,
    var_eps: float = 1e-6,
    clip_std: float = 6.0,
) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    gen_mode = (gen_mode or "gen_var").lower().strip()

    if gen_mode == "gen_var":
        var = np.maximum(stats.var.astype(np.float32), float(var_eps))
        std = np.sqrt(var)
        X = rng.randn(int(n), int(stats.mean.size)).astype(np.float32) * std + stats.mean.astype(np.float32)
        if clip_std is not None and float(clip_std) > 0:
            lo = stats.mean.astype(np.float32) - float(clip_std) * std
            hi = stats.mean.astype(np.float32) + float(clip_std) * std
            X = np.minimum(np.maximum(X, lo), hi)
        return X

    if gen_mode == "gen_min":
        lo = stats.minv.astype(np.float32)
        hi = stats.maxv.astype(np.float32)
        span = (hi - lo)
        span = np.where(np.abs(span) < 1e-12, 1.0, span).astype(np.float32)
        U = rng.rand(int(n), int(stats.mean.size)).astype(np.float32)
        return lo + U * span

    raise ValueError(f"Unknown gen_mode={gen_mode}")


def _train_student_on_xy(student: nn.Module, device: torch.device, Xq_adv: np.ndarray, yT: np.ndarray, cfg: TempestConfig):
    student = student.to(device).train()
    opt = torch.optim.AdamW(student.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    n = int(Xq_adv.shape[0])
    bs = int(cfg.batch_size)

    for ep in range(1, int(cfg.epochs) + 1):
        idx = np.random.permutation(n)
        ep_loss = 0.0
        for si in range(0, n, bs):
            j = idx[si:si + bs]
            x = torch.from_numpy(Xq_adv[j]).to(device).float()
            t = torch.from_numpy(yT[j]).to(device).float()

            opt.zero_grad(set_to_none=True)
            s = student(x).reshape(-1).float()
            loss = F.mse_loss(s, t)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * int(x.size(0))

        ep_loss /= max(n, 1)
        if ep == 1 or ep == int(cfg.epochs) or (ep % 5 == 0):
            print(f"[TEMPEST] epoch {ep}/{cfg.epochs} | mse={ep_loss:.6f} | queries={n}")


def train_student_tempest(
    teacher_score_fn: Callable[[torch.Tensor], torch.Tensor],
    student: nn.Module,
    device: torch.device,
    stats: PublicStats,
    cfg: TempestConfig,
) -> Dict[str, float]:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    Xq = generate_queries_from_stats(
        n=int(cfg.query_budget),
        stats=stats,
        gen_mode=str(cfg.gen_mode),
        seed=int(cfg.seed),
        var_eps=float(cfg.var_eps),
        clip_std=float(cfg.clip_std),
    )

    t0 = time.time()
    with torch.no_grad():
        xb = torch.from_numpy(Xq).to(device).float()
        yT = teacher_score_fn(xb).reshape(-1).detach().cpu().numpy().astype(np.float32)
    t1 = time.time()

    Xq_adv = normalize_adv(Xq, stats=stats, mode=str(cfg.adv_norm)).astype(np.float32, copy=False)
    yT = np.asarray(yT, dtype=np.float32).reshape(-1)

    _train_student_on_xy(student, device, Xq_adv, yT, cfg)

    sec = float(t1 - t0)
    qps = float(cfg.query_budget) / max(sec, 1e-12)
    return dict(
        extract_sec=sec,
        query_budget=int(cfg.query_budget),
        queries_per_sec=qps,
        ms_per_query=1000.0 * sec / max(int(cfg.query_budget), 1),
    )


def train_student_tempest_from_cache(
    Xq_full: np.ndarray,
    yT_full: np.ndarray,
    student: nn.Module,
    device: torch.device,
    stats: PublicStats,
    cfg: TempestConfig,
) -> Dict[str, float]:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    n = int(cfg.query_budget)
    Xq = np.asarray(Xq_full[:n], dtype=np.float32)
    yT = np.asarray(yT_full[:n], dtype=np.float32).reshape(-1)

    Xq_adv = normalize_adv(Xq, stats=stats, mode=str(cfg.adv_norm)).astype(np.float32, copy=False)

    t0 = time.time()
    _train_student_on_xy(student, device, Xq_adv, yT, cfg)
    t1 = time.time()

    sec = float(t1 - t0)
    qps = float(n) / max(sec, 1e-12)
    return dict(
        extract_sec=sec,
        query_budget=int(n),
        queries_per_sec=qps,
        ms_per_query=1000.0 * sec / max(int(n), 1),
    )


__all__ = [
    "TempestConfig",
    "PublicStats",
    "compute_public_stats",
    "generate_queries_from_stats",
    "normalize_adv",
    "train_student_tempest",
    "train_student_tempest_from_cache",
]

