# ============================================================
# FILE: attacks/qselect_dfme.py 
# ============================================================

import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Deque, Callable, List
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.query_selectors.query_selector_gpu_shared_projection import (
    GPUSharedProjCfg,
    GPUSharedProjectionSelector,
)

# -----------------------------
# helpers
# -----------------------------
def _safe_nan_to_num_1d(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s).reshape(-1)
    if np.isfinite(s).all():
        return s.astype(np.float32, copy=False)
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        return np.zeros_like(s, dtype=np.float32)
    hi = float(np.max(finite))
    lo = float(np.min(finite))
    return np.nan_to_num(s, nan=0.0, posinf=hi, neginf=lo).astype(np.float32, copy=False)


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size < 2:
        return float("nan")
    ra = a.argsort().argsort().astype(np.float32)
    rb = b.argsort().argsort().astype(np.float32)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.sqrt((ra * ra).sum()) * np.sqrt((rb * rb).sum())) + 1e-12
    return float((ra * rb).sum() / denom)


# -----------------------------
# generator
# -----------------------------
class MLPGenerator(nn.Module):
    def __init__(self, z_dim: int, d_out: int, hidden=(512, 512), out_act: Optional[str] = None):
        super().__init__()
        layers = []
        prev = z_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, d_out)]
        self.net = nn.Sequential(*layers)
        self.out_act = out_act

    def forward(self, z):
        x = self.net(z)
        if self.out_act is None:
            return x
        if self.out_act == "tanh":
            return torch.tanh(x)
        if self.out_act == "sigmoid":
            return torch.sigmoid(x)
        raise ValueError(self.out_act)


def build_default_generator(d_in: int, z_dim: int = 64, hidden=(512, 512), out_act: Optional[str] = None) -> nn.Module:
    return MLPGenerator(z_dim=z_dim, d_out=d_in, hidden=hidden, out_act=out_act)


# -----------------------------
# OPTIONAL generator helpers (guarded by gen_update_every>0)
# -----------------------------
def _pairwise_diversity_loss(x: torch.Tensor, max_pairs: int = 2048) -> torch.Tensor:
    """
    Encourage diversity by penalizing too-small distances.
    Sample random pairs for speed.
    """
    n = int(x.shape[0])
    if n < 2:
        return torch.tensor(0.0, device=x.device)

    P = int(min(max_pairs, n * (n - 1) // 2))
    if P <= 0:
        return torch.tensor(0.0, device=x.device)

    i = torch.randint(0, n, (P,), device=x.device)
    j = torch.randint(0, n, (P,), device=x.device)
    d = (x[i] - x[j]).pow(2).sum(dim=1).sqrt()
    return torch.relu(0.1 - d).mean()


# -----------------------------
# config
# -----------------------------
@dataclass
class QSelDFMEConfig:
    steps: int = 32
    pool_size: int = 2000
    teacher_batch_size: int = 32
    total_query_budget: int = 1000

    z_dim: int = 64
    lr_student: float = 1e-3
    lr_gen: float = 3e-4
    student_loss: str = "mse_zscore"   # "mse" | "mse_zscore" | "huber"
    lambda_gen: float = 0.2

    use_query_selection: bool = True
    selector_pca_dim: int = 8
    selector_prefilter_ratio: float = 0.25
    selector_candidate_factor: int = 4

    # fraction of K to sample randomly (the rest uses selector)
    random_mix_frac: float = 0.0

    # IMPORTANT: default 0 => generator NOT trained (same behaviour as your file)
    gen_update_every: int = 0
    num_generators: int = 1

    # OPTIONAL gen knobs (only used if gen_update_every>0)
    gen_steps: int = 1
    gen_diversity_w: float = 0.05

    use_feat_bounds: bool = True

    store_calib_buffer: bool = True
    calib_buffer_max: int = 200_000

    student_steps_per_round: int = 4

    replay_ratio: float = 2.0
    replay_max: int = 30_000
    replay_quantiles: int = 5
    replay_cap_mult: float = 3.0      # replay max = cap_mult * K

    rank_loss_weight: float = 0.35
    rank_pairs: int = 2048
    rank_anneal_to: float = 0.10

    refine_x_steps: int = 1
    refine_x_lr: float = 0.04
    refine_x_noise: float = 0.01

    grad_clip: float = 1.0

    min_last_k: int = 16
    teacher_ema_alpha: float = 0.10

    proto_idx: Optional[int] = None
    proto_allowed: Optional[torch.Tensor] = None

    milestones: Optional[List[int]] = None
    milestone_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    # K schedule (trajectory-friendly)
    k_early: int = 32
    k_late: int = 2048
    k_switch_at: int = 10_000

    seed: int = 42
    log_every: int = 10


# -----------------------------
# losses
# -----------------------------
def _student_loss(pred: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if mode == "mse":
        return F.mse_loss(pred, target)
    if mode == "huber":
        return F.smooth_l1_loss(pred, target)
    if mode == "mse_zscore":
        mu = target.mean()
        sd = target.std().clamp_min(1e-6)
        return F.mse_loss((pred - mu) / sd, (target - mu) / sd)
    raise ValueError(f"Unknown student_loss={mode}")


def _pairwise_rank_loss(student: torch.Tensor, teacher: torch.Tensor, num_pairs: int) -> torch.Tensor:
    s = student.reshape(-1)
    t = teacher.reshape(-1)
    n = int(s.numel())
    if n < 2:
        return torch.tensor(0.0, device=s.device)

    P = int(min(max(0, num_pairs), n * (n - 1)))
    if P <= 0:
        return torch.tensor(0.0, device=s.device)

    i = torch.randint(0, n, (P,), device=s.device)
    j = torch.randint(0, n, (P,), device=s.device)
    ti, tj = t[i], t[j]
    si, sj = s[i], s[j]

    sign = torch.sign(ti - tj)
    mask = (sign != 0)
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=s.device)

    d = (si - sj)[mask]
    sign = sign[mask]
    return F.softplus(-sign * d).mean()


# -----------------------------
# refinement / bounds
# -----------------------------
def _apply_bounds(x: torch.Tensor, feat_lo: torch.Tensor, feat_hi: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(x, feat_hi), feat_lo)


def _snap_discrete_feature(x: torch.Tensor, feat_idx: int, allowed: torch.Tensor) -> torch.Tensor:
    if allowed is None or allowed.numel() == 0:
        return x
    v = x[:, feat_idx:feat_idx + 1]
    dist = torch.abs(v - allowed.view(1, -1))
    nn_idx = torch.argmin(dist, dim=1)
    x[:, feat_idx] = allowed[nn_idx]
    return x


def _refine_queries_to_boundary(student: nn.Module,
                               x: torch.Tensor,
                               steps: int,
                               lr: float,
                               noise: float,
                               feat_lo: torch.Tensor,
                               feat_hi: torch.Tensor) -> torch.Tensor:
    if steps <= 0:
        return x

    x_ref = x.detach().clone()
    x_ref.requires_grad_(True)

    for _ in range(int(steps)):
        s = student(x_ref).reshape(-1)
        z = (s - s.mean()) / (s.std().clamp_min(1e-6))
        loss = torch.mean(torch.abs(z))
        g = torch.autograd.grad(loss, x_ref, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            x_ref -= float(lr) * g
            if noise and float(noise) > 0:
                x_ref += float(noise) * torch.randn_like(x_ref)
            x_ref[:] = _apply_bounds(x_ref, feat_lo, feat_hi)

    return x_ref.detach()


# -----------------------------
# pool generation
# -----------------------------
def _gen_pool(generator: nn.Module,
              z_dim: int,
              pool_size: int,
              device,
              feat_lo: torch.Tensor,
              feat_hi: torch.Tensor,
              use_feat_bounds: bool) -> torch.Tensor:
    z = torch.randn(pool_size, z_dim, device=device)
    x = generator(z)
    if use_feat_bounds:
        x = _apply_bounds(x, feat_lo, feat_hi)
    return x


def _build_stats_snapshot(
    total_budget: int,
    used: int,
    t0: float,
    cfg: QSelDFMEConfig,
    calib_X: list,
    calib_y: list,
    calib_max: int,
) -> Dict[str, Any]:
    extract_sec = float(time.time() - t0)
    qps = float(used / max(extract_sec, 1e-9))
    ms_per_q = float(1000.0 * extract_sec / max(used, 1))
    stats: Dict[str, Any] = dict(
        query_budget_expected=int(total_budget),
        query_budget=int(used),
        extract_sec=float(extract_sec),
        queries_per_sec=float(qps),
        ms_per_query=float(ms_per_q),
    )

    if bool(cfg.store_calib_buffer) and calib_X and calib_y:
        Xc = np.concatenate(calib_X, axis=0)
        yc = np.concatenate(calib_y, axis=0)
        if Xc.shape[0] > calib_max:
            Xc = Xc[-calib_max:]
            yc = yc[-calib_max:]
        stats["calib_X_np"] = Xc.astype(np.float32, copy=False)
        stats["calib_y_np"] = _safe_nan_to_num_1d(yc)

    return stats


# -----------------------------
# main training
# -----------------------------
def train_student_qselect_dfme(
    teacher_score_fn,
    student: nn.Module,
    generator: nn.Module,
    device,
    cfg: QSelDFMEConfig,
    feat_lo: torch.Tensor,
    feat_hi: torch.Tensor,
) -> Dict[str, Any]:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    student.train()
    generator.train()

    opt_s = torch.optim.Adam(student.parameters(), lr=float(cfg.lr_student))

    # OPTIONAL: generator optimizer (OFF by default)
    opt_g = None
    if int(getattr(cfg, "gen_update_every", 0)) > 0:
        opt_g = torch.optim.Adam(generator.parameters(), lr=float(cfg.lr_gen))

    selector = GPUSharedProjectionSelector(
        d_in=int(feat_lo.numel()),
        cfg=GPUSharedProjCfg(
            dim=int(cfg.selector_pca_dim),
            seed=int(cfg.seed),
            prefilter_ratio=float(cfg.selector_prefilter_ratio),
            candidate_factor=int(cfg.selector_candidate_factor),
            use_cosine=True,
        ),
        device=device,
    )

    total_budget = int(cfg.total_query_budget)
    steps = int(max(1, cfg.steps))

    used = 0
    t0 = time.time()

    replay_x: Deque[torch.Tensor] = deque(maxlen=int(cfg.replay_max))
    replay_y: Deque[torch.Tensor] = deque(maxlen=int(cfg.replay_max))

    calib_X = []
    calib_y = []
    calib_max = int(cfg.calib_buffer_max)

    ema_mu = None
    ema_sd = None
    alpha = float(cfg.teacher_ema_alpha)

    milestones = None
    if cfg.milestones is not None:
        milestones = sorted([int(m) for m in cfg.milestones if int(m) > 0])

    for step_i in range(steps):
        remaining = total_budget - used
        if remaining <= 0:
            break

        # K schedule
        k_target = int(cfg.k_early) if used < int(cfg.k_switch_at) else int(cfg.k_late)
        K = int(min(k_target, remaining))
        if K <= 0:
            break

        x_pool = _gen_pool(
            generator=generator,
            z_dim=int(cfg.z_dim),
            pool_size=int(cfg.pool_size),
            device=device,
            feat_lo=feat_lo,
            feat_hi=feat_hi,
            use_feat_bounds=bool(cfg.use_feat_bounds),
        )

        with torch.no_grad():
            s_pool = student(x_pool).reshape(-1)

        if bool(cfg.use_query_selection):
            idx_t = selector.select(x_pool, k=K, student_scores=s_pool)
        else:
            idx_t = torch.randperm(x_pool.shape[0], device=device)[:K]

        # -----------------------------
        # random-mix for diversity
        # -----------------------------
        mix = float(getattr(cfg, "random_mix_frac", 0.0))
        if mix > 0 and K >= 2:
            m = int(round(mix * K))
            m = max(0, min(m, K))
            if m > 0:
                keep = K - m
                idx_keep = idx_t[:keep]

                poolN = int(x_pool.shape[0])
                rnd = torch.randperm(poolN, device=device)[: (m * 3)]
                if keep > 0:
                    mask = torch.ones(rnd.numel(), device=device, dtype=torch.bool)
                    for v in idx_keep:
                        mask &= (rnd != v)
                    rnd = rnd[mask]
                if rnd.numel() < m:
                    rnd = torch.randperm(poolN, device=device)[:m]
                idx_rand = rnd[:m]
                idx_t = torch.cat([idx_keep, idx_rand], dim=0)

        x_q = x_pool.index_select(0, idx_t)

        if int(cfg.refine_x_steps) > 0:
            x_q = _refine_queries_to_boundary(
                student=student,
                x=x_q,
                steps=int(cfg.refine_x_steps),
                lr=float(cfg.refine_x_lr),
                noise=float(cfg.refine_x_noise),
                feat_lo=feat_lo,
                feat_hi=feat_hi,
            )

        if cfg.proto_idx is not None and cfg.proto_allowed is not None:
            x_q = _snap_discrete_feature(x_q, int(cfg.proto_idx), cfg.proto_allowed)

        # Teacher query (counts toward budget)
        y_q = teacher_score_fn(x_q).detach().reshape(-1)
        used += int(x_q.shape[0])

        # -----------------------------
        # OPTIONAL generator training (OFF by default)
        # Budget-safe: does NOT call teacher again.
        # -----------------------------
        if opt_g is not None:
            upd_every = int(getattr(cfg, "gen_update_every", 0))
            do_update = (upd_every > 0) and (((step_i + 1) % upd_every) == 0)
            if do_update:
                gen_steps = int(getattr(cfg, "gen_steps", 1))
                div_w = float(getattr(cfg, "gen_diversity_w", 0.05))

                with torch.no_grad():
                    s_q = student(x_q).reshape(-1)

                for _ in range(max(1, gen_steps)):
                    z_g = torch.randn(int(x_q.shape[0]), int(cfg.z_dim), device=device)
                    x_g = generator(z_g)
                    if bool(cfg.use_feat_bounds):
                        x_g = _apply_bounds(x_g, feat_lo, feat_hi)
                    if cfg.proto_idx is not None and cfg.proto_allowed is not None:
                        x_g = _snap_discrete_feature(x_g, int(cfg.proto_idx), cfg.proto_allowed)

                    s_g = student(x_g).reshape(-1)

                    # match score mean/std to informative batch (stable)
                    loss_match = (
                        F.mse_loss(s_g.mean(), s_q.mean()) +
                        F.mse_loss(s_g.std().clamp_min(1e-6), s_q.std().clamp_min(1e-6))
                    )
                    loss_div = _pairwise_diversity_loss(x_g, max_pairs=2048)
                    loss_g = loss_match + div_w * loss_div

                    opt_g.zero_grad(set_to_none=True)
                    loss_g.backward()
                    if float(cfg.grad_clip) and float(cfg.grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(generator.parameters(), float(cfg.grad_clip))
                    opt_g.step()

        x_q = x_q.detach()
        y_q = y_q.detach()

        # EMA scale update
        with torch.no_grad():
            mu = y_q.mean()
            sd = y_q.std().clamp_min(1e-6)
            if ema_mu is None:
                ema_mu = mu
                ema_sd = sd
            else:
                ema_mu = (1.0 - alpha) * ema_mu + alpha * mu
                ema_sd = (1.0 - alpha) * ema_sd + alpha * sd

        replay_x.append(x_q.cpu())
        replay_y.append(y_q.cpu())

        if bool(cfg.store_calib_buffer):
            calib_X.append(x_q.cpu().numpy().astype(np.float32, copy=False))
            calib_y.append(y_q.cpu().numpy().astype(np.float32, copy=False))
            if sum(a.shape[0] for a in calib_y) > calib_max:
                while calib_X and sum(a.shape[0] for a in calib_y) > calib_max:
                    calib_X.pop(0)
                    calib_y.pop(0)

        # student step schedule
        if total_budget <= 200:
            stu_steps = max(int(cfg.student_steps_per_round), 6)
        elif total_budget <= 1000:
            stu_steps = max(int(cfg.student_steps_per_round), 4)
        else:
            stu_steps = max(int(cfg.student_steps_per_round), 2)

        # rank anneal
        t = step_i / float(max(1, steps - 1))
        rank_w = float(cfg.rank_loss_weight) * ((1.0 - t) + float(cfg.rank_anneal_to) * t)

        replay_bs = int(round(float(cfg.replay_ratio) * K))
        replay_cap = int(round(float(cfg.replay_cap_mult) * K))
        replay_bs = max(0, min(replay_bs, replay_cap))

        last_loss = None
        for _ in range(int(stu_steps)):
            xb = x_q
            yb = y_q

            if replay_bs > 0 and int(cfg.replay_quantiles) > 0 and len(replay_x) > 1:
                chunks = list(replay_x)[-min(8, len(replay_x)):]
                chunky = list(replay_y)[-min(8, len(replay_y)):]
                Xr = torch.cat(chunks, dim=0)
                Yr = torch.cat(chunky, dim=0)

                y_cpu = Yr.reshape(-1)
                qs = torch.linspace(0, 1, int(cfg.replay_quantiles) + 1)
                edges = torch.quantile(y_cpu, qs)

                take_per = max(1, int(replay_bs / int(cfg.replay_quantiles)))
                idxs = []
                for qi in range(int(cfg.replay_quantiles)):
                    lo = edges[qi]
                    hi = edges[qi + 1]
                    m = (y_cpu >= lo) & (y_cpu <= hi)
                    cand = torch.where(m)[0]
                    if cand.numel() > 0:
                        take = min(take_per, int(cand.numel()))
                        sel = cand[torch.randperm(int(cand.numel()))[:take]]
                        idxs.append(sel)

                if idxs:
                    idxs = torch.cat(idxs, dim=0)
                    if idxs.numel() > replay_bs:
                        idxs = idxs[torch.randperm(int(idxs.numel()))[:replay_bs]]

                    Xrb = Xr.index_select(0, idxs).to(device, non_blocking=True)
                    Yrb = Yr.index_select(0, idxs).to(device, non_blocking=True)
                    xb = torch.cat([xb, Xrb], dim=0)
                    yb = torch.cat([yb, Yrb], dim=0)

            pred = student(xb).reshape(-1)
            if ema_mu is not None and ema_sd is not None and cfg.student_loss == "mse_zscore":
                loss_fit = F.mse_loss((pred - ema_mu) / ema_sd, (yb - ema_mu) / ema_sd)
            else:
                loss_fit = _student_loss(pred, yb, mode=str(cfg.student_loss))

            loss_rank = _pairwise_rank_loss(pred, yb, num_pairs=int(cfg.rank_pairs))
            loss = loss_fit + float(rank_w) * loss_rank

            opt_s.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg.grad_clip) and float(cfg.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg.grad_clip))
            opt_s.step()
            last_loss = loss

        if int(cfg.log_every) > 0 and ((step_i + 1) % int(cfg.log_every) == 0 or used >= total_budget):
            lv = float(last_loss.item()) if last_loss is not None else float("nan")
            print(f"[QSelDFME] step={step_i+1:>3}/{steps} K={K:>5} pool={int(cfg.pool_size):>5} used={used:>8}/{total_budget} loss={lv:.6f}")

        # milestone callback
        if milestones and cfg.milestone_callback:
            while milestones and used >= int(milestones[0]):
                snap = _build_stats_snapshot(
                    total_budget=total_budget,
                    used=used,
                    t0=t0,
                    cfg=cfg,
                    calib_X=calib_X,
                    calib_y=calib_y,
                    calib_max=calib_max,
                )
                cfg.milestone_callback({"used": int(used), "stats": snap})
                milestones.pop(0)

    stats = _build_stats_snapshot(
        total_budget=total_budget,
        used=used,
        t0=t0,
        cfg=cfg,
        calib_X=calib_X,
        calib_y=calib_y,
        calib_max=calib_max,
    )
    return stats


# -----------------------------
# fidelity eval
# -----------------------------
def score_fidelity_on_loader(teacher_score_fn, student: nn.Module, loader, device) -> Dict[str, float]:
    student.eval()
    t_all = []
    s_all = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            t = teacher_score_fn(xb).detach().reshape(-1).cpu().numpy()
            s = student(xb).detach().reshape(-1).cpu().numpy()
            t_all.append(t)
            s_all.append(s)

    t_all = _safe_nan_to_num_1d(np.concatenate(t_all, axis=0))
    s_all = _safe_nan_to_num_1d(np.concatenate(s_all, axis=0))

    mse = float(np.mean((t_all - s_all) ** 2))
    mae = float(np.mean(np.abs(t_all - s_all)))
    rho = float(_spearman_rho(t_all, s_all))
    return {"score_mse": mse, "score_mae": mae, "spearman_rho": rho}


# -----------------------------
# calibration
# -----------------------------
class AffineCalibratedStudent(nn.Module):
    def __init__(self, base: nn.Module, a: float, b: float):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.tensor(float(a), dtype=torch.float32), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(float(b), dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        y = self.base(x).reshape(-1)
        return (self.a * y + self.b).reshape(-1)


def calibrate_student_affine(
    teacher_score_fn,
    student: nn.Module,
    d_in: int,
    device,
    feat_lo: torch.Tensor,
    feat_hi: torch.Tensor,
    calib_X: torch.Tensor,
    calib_y: torch.Tensor,
    force_monotonic: bool = True,
) -> Dict[str, Any]:
    student.eval()
    with torch.no_grad():
        X = calib_X.to(device).float()
        if X.ndim == 1:
            X = X.view(1, -1)
        if X.shape[1] != d_in:
            X = X[:, :d_in]
        if X.numel() == 0:
            return {"a": 1.0, "b": 0.0, "calibrated_student": student}

        ys = student(X).detach().reshape(-1).float()
        yt = calib_y.to(device).detach().reshape(-1).float()
        ys_np = ys.cpu().numpy().astype(np.float64)
        yt_np = yt.cpu().numpy().astype(np.float64)

    A = np.stack([ys_np, np.ones_like(ys_np)], axis=1)
    sol, _, _, _ = np.linalg.lstsq(A, yt_np, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    if force_monotonic and a < 0.0:
        a = abs(a)

    cal_student = AffineCalibratedStudent(student, a=a, b=b).to(device)
    return {"a": a, "b": b, "calibrated_student": cal_student}


__all__ = [
    "QSelDFMEConfig",
    "train_student_qselect_dfme",
    "score_fidelity_on_loader",
    "calibrate_student_affine",
    "build_default_generator",
]

