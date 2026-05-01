# ============================================================
# FILE: experiments/utils/fidelity.py


from typing import Callable, Dict
import numpy as np
import torch
import torch.nn as nn


def safe_nan_to_num_1d(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s).reshape(-1)
    if np.isfinite(s).all():
        return s.astype(np.float32, copy=False)
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        return np.zeros_like(s, dtype=np.float32)
    hi = float(np.max(finite))
    lo = float(np.min(finite))
    return np.nan_to_num(s, nan=0.0, posinf=hi, neginf=lo).astype(np.float32, copy=False)


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    # EXACTLY the same as attacks/qselect_dfme.py::_spearman_rho
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


@torch.no_grad()
def score_fidelity_on_loader(
    teacher_score_fn: Callable[[torch.Tensor], torch.Tensor],
    student: nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    student = student.to(device).eval()

    t_all, s_all = [], []
    for batch in loader:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        xb = xb.to(device).float()

        t = teacher_score_fn(xb).detach().reshape(-1).cpu().numpy()
        s = student(xb).detach().reshape(-1).cpu().numpy()
        t_all.append(t)
        s_all.append(s)

    t_all = safe_nan_to_num_1d(np.concatenate(t_all, axis=0))
    s_all = safe_nan_to_num_1d(np.concatenate(s_all, axis=0))

    mse = float(np.mean((t_all - s_all) ** 2))
    mae = float(np.mean(np.abs(t_all - s_all)))
    rho = float(spearman_rho(t_all, s_all))

    return {"score_mse": mse, "score_mae": mae, "spearman_rho": rho}