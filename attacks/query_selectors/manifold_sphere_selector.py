# ============================================================
# FILE: attacks/query_selector_manifold_shared_projection.py
# ============================================================


import numpy as np
from dataclasses import dataclass
from typing import Optional

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import MiniBatchKMeans


def _safe_np(X: np.ndarray) -> np.ndarray:
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


def _zscore_1d(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    mu = float(np.mean(s)) if s.size else 0.0
    sd = float(np.std(s)) if s.size else 1.0
    sd = sd if sd > 1e-6 else 1.0
    return (s - mu) / sd


def _farthest_first(X: np.ndarray, k: int, start_idx: int = 0) -> np.ndarray:
    n = X.shape[0]
    if k <= 0:
        return np.array([], dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)

    sel = [int(start_idx)]
    d2 = np.sum((X - X[start_idx]) ** 2, axis=1)

    for _ in range(1, k):
        i = int(np.argmax(d2))
        sel.append(i)
        d2 = np.minimum(d2, np.sum((X - X[i]) ** 2, axis=1))
    return np.array(sel, dtype=int)


def _coverage_pick(X: np.ndarray, k: int, n_clusters: Optional[int], seed: int) -> np.ndarray:
    n = X.shape[0]
    if k <= 0:
        return np.array([], dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)

    if n_clusters is None:
        n_clusters = min(max(10, k * 4), 200)

    km = MiniBatchKMeans(
        n_clusters=int(n_clusters),
        random_state=int(seed),
        batch_size=4096,
        n_init=10,  # IMPORTANT: int for sklearn compatibility
    ).fit(X)

    labels = km.labels_
    centers = km.cluster_centers_
    dist2 = np.sum((X - centers[labels]) ** 2, axis=1)

    rng = np.random.RandomState(seed)
    picked = []
    for c in rng.permutation(int(n_clusters)):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        picked.append(int(idx[np.argmin(dist2[idx])]))
        if len(picked) >= k:
            break

    if len(picked) < k:
        remaining = np.setdiff1d(np.arange(n), np.array(picked, dtype=int), assume_unique=False)
        extra = _farthest_first(X[remaining], k - len(picked), start_idx=0)
        picked.extend(list(remaining[extra]))

    return np.array(picked, dtype=int)


@dataclass
class SharedProjectionCfg:
    method: str = "pca"     # "pca" | "rp" | "none"
    dim: int = 8
    whiten: bool = False
    standardize: bool = True
    seed: int = 0


class SharedProjector:
    def __init__(self, cfg: SharedProjectionCfg):
        self.cfg = cfg
        self.scaler: Optional[StandardScaler] = None
        self.proj = None
        self.fitted = False

    def fit(self, X: np.ndarray):
        X = _safe_np(X)
        Xf = X
        if self.cfg.standardize:
            self.scaler = StandardScaler(with_mean=True, with_std=True)
            Xf = self.scaler.fit_transform(Xf)

        if self.cfg.method == "none":
            self.proj = None
        elif self.cfg.method == "pca":
            d = int(min(self.cfg.dim, Xf.shape[1]))
            self.proj = PCA(n_components=d, whiten=self.cfg.whiten, random_state=self.cfg.seed)
            self.proj.fit(Xf)
        elif self.cfg.method == "rp":
            d = int(min(self.cfg.dim, Xf.shape[1]))
            self.proj = GaussianRandomProjection(n_components=d, random_state=self.cfg.seed)
            self.proj.fit(Xf)
        else:
            raise ValueError(f"Unknown projection method: {self.cfg.method}")

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("SharedProjector not fitted")
        X = _safe_np(X)
        Xf = X
        if self.scaler is not None:
            Xf = self.scaler.transform(Xf)
        if self.proj is None:
            return Xf
        return self.proj.transform(Xf)


@dataclass
class ManifoldSelectorCfg:
    phaseA_frac: float = 0.45
    candidate_factor: int = 5
    candidate_min: int = 200
    prefilter_ratio: float = 0.10
    n_clusters: Optional[int] = None
    seed: int = 0


class ManifoldSharedProjectionQuerySelector:
    """
    Score-teacher selector.
    Uncertainty proxy: high when |zscore(student(x))| is small (near boundary).
    """
    def __init__(self, proj_cfg: SharedProjectionCfg, sel_cfg: ManifoldSelectorCfg):
        self.proj_cfg = proj_cfg
        self.sel_cfg = sel_cfg
        self.projector = SharedProjector(proj_cfg)
        self._has_fit = False

    def maybe_refit(self, X_pool: np.ndarray, force: bool = False):
        if force or (not self._has_fit):
            self.projector.fit(X_pool)
            self._has_fit = True

    def select(
        self,
        X_pool: np.ndarray,
        k: int,
        student_scores: Optional[np.ndarray],
        use_uncertainty: bool,
        uncertainty_top_ratio: float,
    ) -> np.ndarray:
        X_pool = _safe_np(X_pool)
        n = X_pool.shape[0]
        k = int(min(max(0, k), n))
        if k <= 0:
            return np.array([], dtype=int)
        if k >= n:
            return np.arange(n, dtype=int)

        self.maybe_refit(X_pool, force=False)
        X_emb = self.projector.transform(X_pool)

        pool_idx = np.arange(n)

        # --- uncertainty proxy for scores: boundary = small |z|
        if use_uncertainty and student_scores is not None:
            z = _zscore_1d(student_scores)
            uncert = np.exp(-np.abs(z))  # high near 0
        else:
            uncert = None

        # --- prefilter (keep high-uncertainty candidates)
        if uncert is not None and 0.0 < float(self.sel_cfg.prefilter_ratio) < 1.0:
            m = int(max(k * 5, round(n * float(self.sel_cfg.prefilter_ratio))))
            m = min(m, n)
            keep = np.argsort(-uncert)[:m]
            pool_idx = pool_idx[keep]
            X_emb_use = X_emb[keep]
            uncert_use = uncert[keep]
        else:
            X_emb_use = X_emb
            uncert_use = uncert

        # --- Phase A coverage
        kA = int(min(k, max(1, round(k * float(self.sel_cfg.phaseA_frac)))))
        idxA_local = _coverage_pick(
            X_emb_use,
            k=kA,
            n_clusters=self.sel_cfg.n_clusters,
            seed=self.sel_cfg.seed,
        )
        idxA = pool_idx[idxA_local]

        if kA == k:
            return idxA

        # --- Phase B informative + diverse
        remaining = np.setdiff1d(pool_idx, idxA, assume_unique=False)
        if remaining.size == 0:
            return idxA

        Xr = X_emb[remaining]
        kB = k - kA

        if uncert is None:
            pick_local = _farthest_first(Xr, kB, start_idx=0)
            idxB = remaining[pick_local]
            return np.concatenate([idxA, idxB], axis=0)

        # ===== IMPORTANT FIX: respect uncertainty_top_ratio =====
        ur = uncert[remaining]

        base_top = int(min(
            max(self.sel_cfg.candidate_factor * kB, self.sel_cfg.candidate_min),
            remaining.size
        ))

        r = float(uncertainty_top_ratio)
        if 0.0 < r < 1.0:
            top_cap = int(max(kB, round(r * remaining.size)))
            base_top = int(min(base_top, top_cap))

        top = int(max(kB, base_top))
        top = int(min(top, remaining.size))
        # =======================================================

        cand_local = np.argsort(-ur)[:top]
        cand_idx = remaining[cand_local]
        Xcand = X_emb[cand_idx]

        start = int(np.random.RandomState(self.sel_cfg.seed).randint(0, Xcand.shape[0]))
        pick_local = _farthest_first(Xcand, kB, start_idx=start)
        idxB = cand_idx[pick_local]

        return np.concatenate([idxA, idxB], axis=0)