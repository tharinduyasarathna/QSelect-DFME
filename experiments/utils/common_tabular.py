# ============================================================
# FILE: experiments/utils/common_tabular.py
# ============================================================

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import TensorDataset, DataLoader


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    X_t = torch.from_numpy(X.astype(np.float32, copy=False))
    y_t = torch.from_numpy(y.astype(np.int64, copy=False))
    ds = TensorDataset(X_t, y_t)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def stratified_subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int):
    if n is None or int(n) <= 0 or int(n) >= int(X.shape[0]):
        return X, y

    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    n = int(n)

    classes, counts = np.unique(y, return_counts=True)
    frac = n / float(X.shape[0])

    idx_all = []
    for c, cnt in zip(classes, counts):
        idx_c = np.where(y == c)[0]
        take = max(1, int(round(frac * cnt)))
        take = min(take, idx_c.size)
        idx_all.append(rng.choice(idx_c, size=take, replace=False))

    idx = np.concatenate(idx_all)
    if idx.size > n:
        idx = rng.choice(idx, size=n, replace=False)
    elif idx.size < n:
        remaining = np.setdiff1d(np.arange(X.shape[0]), idx, assume_unique=False)
        if remaining.size > 0:
            add = rng.choice(remaining, size=min(n - idx.size, remaining.size), replace=False)
            idx = np.concatenate([idx, add])

    rng.shuffle(idx)
    return X[idx], y[idx]


def sanitize_np(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X).all():
        X = X.copy()
        X[~np.isfinite(X)] = np.nan
    if np.isnan(X).any():
        X = X.copy()
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        inds = np.where(np.isnan(X))
        X[inds] = col_means[inds[1]]
    return X


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


def safe_auc(y_true_bin: np.ndarray, scores: np.ndarray):
    u = np.unique(y_true_bin)
    if len(u) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true_bin, scores)), float(average_precision_score(y_true_bin, scores))


def safe_slug(s: str) -> str:
    return "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in s])


def infer_protocol_allowed_from_scaled_train(X_train: np.ndarray, proto_idx: int, device) -> torch.Tensor:
    col = X_train[:, int(proto_idx)]
    col = col[np.isfinite(col)]
    if col.size == 0:
        return torch.tensor([0.0], device=device)

    uniq = np.unique(col)
    if uniq.size <= 10:
        return torch.tensor(uniq.astype(np.float32), device=device)

    qs = np.quantile(col, [0.0, 0.5, 1.0]).astype(np.float32)
    qs = np.unique(qs)
    return torch.tensor(qs, device=device)