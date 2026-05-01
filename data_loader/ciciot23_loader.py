# ============================================================
# FILE: data_loader/ciciot23_loader.py  


import os
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import TensorDataset, DataLoader

from .base_tabular import BaseTabularConfig, BaseTabularDataModule


# -----------------------------
# Optional: priority features (appear first if present)
# (NOT restrictive; missing columns are ignored)
# -----------------------------
DEFAULT_PRIORITY_FEATURES: List[str] = [
    "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count",
    "Data_length",  # may not exist in your dump; safe
    "Weight",
    "AVG", "Std", "Tot sum", "Tot size", "IAT", "Magnitue",
]

# -----------------------------
# Optional: columns that are often redundant / shortcut-ish.
# You can disable by setting drop_cols=[] in config.
# -----------------------------
DEFAULT_DROP_COLS: List[str] = [
    # redundant aggregate stats (you already flagged)
    "Variance", "Covariance", "Radius",

    # sometimes too dataset-specific / shortcut
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
]

DEFAULT_BENIGN_TOKENS = {"BENIGN", "NORMAL", "0","BenignTraffic"}


def _norm_str_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip().str.upper()
    out = out.str.replace(r"\s+", " ", regex=True)
    return out


@dataclass
class CICIoT23Config(BaseTabularConfig):
    csv_dir: str = ""
    files: Optional[Sequence[str]] = None
    label_col: str = "label"

    # Keep all useful cols by default
    drop_cols: Optional[List[str]] = None
    drop_constant_cols: bool = True

    # Optional: reorder columns so these come first (if present)
    priority_features: Optional[Sequence[str]] = None
    reorder_by_priority: bool = True

    # binary only
    task: str = "binary"

    # label handling
    benign_tokens: Optional[Sequence[str]] = None


class CICIoT23DataModule(BaseTabularDataModule):
    def __init__(self, cfg: CICIoT23Config):
        super().__init__(cfg)
        self.cfg: CICIoT23Config = cfg
        self.label_to_id: Dict[str, int] = {"BENIGN": 0, "ATTACK": 1}

        scale = str(cfg.scale).lower().strip()
        if scale == "zscore":
            self.scaler = StandardScaler(with_mean=True, with_std=True)
        elif scale in ("qt_minmax", "qt+minmax", "minmax_01"):
            self.scaler = Pipeline([
                ("qt", QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=1000,
                    subsample=200_000,
                    random_state=cfg.random_state
                )),
                ("mm", MinMaxScaler(feature_range=(0.0, 1.0))),
            ])
        elif scale == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown scale='{cfg.scale}'. Use: zscore | qt_minmax | none")

    def dataset_name(self) -> str:
        return "CICIoT23"

    # -----------------------------
    # IO
    # -----------------------------
    def _resolve_files(self) -> List[str]:
        if self.cfg.files is not None:
            paths = [os.path.join(self.cfg.csv_dir, f) for f in self.cfg.files]
            paths = [p for p in paths if os.path.exists(p)]
            if not paths:
                raise FileNotFoundError(
                    f"No specified CICIoT23 files found in {self.cfg.csv_dir}. "
                    f"Tried: {list(self.cfg.files)}"
                )
            return paths

        paths = sorted(glob.glob(os.path.join(self.cfg.csv_dir, "*.csv")))
        if not paths:
            raise FileNotFoundError(f"No *.csv files found in: {self.cfg.csv_dir}")
        return paths

    def _load_raw_df(self) -> pd.DataFrame:
        dfs = []
        for p in self._resolve_files():
            dfs.append(pd.read_csv(p, low_memory=False))
        return pd.concat(dfs, axis=0, ignore_index=True)

    # -----------------------------
    # Stratified subsample AFTER cleaning
    # -----------------------------
    def _stratified_subsample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if n is None or len(df) <= n:
            return df

        rng = np.random.RandomState(self.cfg.random_state)
        y = df[self.cfg.label_col].astype(int)
        vc = y.value_counts()
        labels = vc.index.tolist()
        counts = vc.values.astype(int)

        alloc = np.floor(counts / counts.sum() * n).astype(int)
        alloc = np.maximum(alloc, 1)

        diff = int(n) - int(alloc.sum())
        if diff > 0:
            order = np.argsort(-counts)
            for i in range(diff):
                alloc[order[i % len(order)]] += 1
        elif diff < 0:
            order = np.argsort(-alloc)
            for i in range(-diff):
                j = order[i % len(order)]
                if alloc[j] > 1:
                    alloc[j] -= 1

        parts = []
        for lab, take in zip(labels, alloc):
            sub = df[df[self.cfg.label_col] == lab]
            if len(sub) <= int(take):
                parts.append(sub)
            else:
                idx = rng.choice(len(sub), size=int(take), replace=False)
                parts.append(sub.iloc[idx])

        out = pd.concat(parts, axis=0, ignore_index=True)
        out = out.sample(frac=1.0, random_state=self.cfg.random_state).reset_index(drop=True)
        return out

    # -----------------------------
    # Cleaning
    # -----------------------------
    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        # label col fallback
        if self.cfg.label_col not in df.columns:
            for alt in ("Label", "LABEL", "target", "Target"):
                if alt in df.columns:
                    self.cfg.label_col = alt
                    break
        if self.cfg.label_col not in df.columns:
            raise KeyError(f"Label column '{self.cfg.label_col}' not found.")

        # drop configured columns (never drop label)
        drop_cols = self.cfg.drop_cols if self.cfg.drop_cols is not None else DEFAULT_DROP_COLS
        drop_cols = [c for c in drop_cols if c in df.columns and c != self.cfg.label_col]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # replace inf strings/values early
        df = df.replace([np.inf, -np.inf, "Infinity", "inf", "-inf"], np.nan)

        # label -> binary int (0 benign, 1 attack)
        y_raw = _norm_str_series(df[self.cfg.label_col])
        benign_tokens = set(_norm_str_series(pd.Series(list(
            self.cfg.benign_tokens if self.cfg.benign_tokens is not None else DEFAULT_BENIGN_TOKENS
        ))).tolist())

        benign_mask = y_raw.isin(benign_tokens)
        df[self.cfg.label_col] = (~benign_mask).astype(int)

        # Convert all non-label columns to numeric (coerce errors)
        feat_cols = [c for c in df.columns if c != self.cfg.label_col]
        for c in feat_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Keep only numeric features that exist now
        feat_cols = [c for c in df.columns if c != self.cfg.label_col]

        # Fill NaNs: median then 0.0 for all-NaN cols
        med = df[feat_cols].median(numeric_only=True)
        df[feat_cols] = df[feat_cols].fillna(med)
        df[feat_cols] = df[feat_cols].fillna(0.0)

        # Optionally drop constant cols
        if self.cfg.drop_constant_cols and len(feat_cols) > 0:
            nunique = df[feat_cols].nunique(dropna=False)
            const_cols = nunique[nunique <= 1].index.tolist()
            if const_cols:
                df = df.drop(columns=const_cols)

        # Optionally reorder: priority first, then rest
        if self.cfg.reorder_by_priority:
            priority = list(self.cfg.priority_features) if self.cfg.priority_features is not None else list(DEFAULT_PRIORITY_FEATURES)
            present_pri = [c for c in priority if c in df.columns and c != self.cfg.label_col]
            rest = [c for c in df.columns if c not in present_pri and c != self.cfg.label_col]
            df = df[present_pri + rest + [self.cfg.label_col]]

        # final finite safety
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df

    # -----------------------------
    # Main
    # -----------------------------
    def prepare(self) -> None:
        df = self._load_raw_df()
        df = self._clean_df(df)

        if self.cfg.max_rows is not None:
            df = self._stratified_subsample(df, int(self.cfg.max_rows))

        y = df[self.cfg.label_col].astype(int).to_numpy(dtype=np.int64)
        X = df.drop(columns=[self.cfg.label_col]).to_numpy(dtype=np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y
        )

        # rebuild qt_minmax safely
        if isinstance(self.scaler, Pipeline):
            scale = str(self.cfg.scale).lower().strip()
            if scale in ("qt_minmax", "qt+minmax", "minmax_01"):
                n_train = int(X_train.shape[0])
                safe_nq = int(min(1000, max(10, n_train - 1)))
                self.scaler = Pipeline([
                    ("qt", QuantileTransformer(
                        output_distribution="normal",
                        n_quantiles=safe_nq,
                        subsample=min(200_000, n_train),
                        random_state=self.cfg.random_state
                    )),
                    ("mm", MinMaxScaler(feature_range=(0.0, 1.0))),
                ])

        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train).astype(np.float32)
            X_test = self.scaler.transform(X_test).astype(np.float32)

        # PyOD-safe
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        self.X_train, self.y_train = X_train, y_train.astype(np.int64)
        self.X_test, self.y_test = X_test, y_test.astype(np.int64)

        # class weights
        classes = np.unique(self.y_train)
        if len(classes) >= 2:
            w = compute_class_weight(class_weight="balanced", classes=classes, y=self.y_train)
            w_full = np.zeros((2,), dtype=np.float32)
            for cid, ww in zip(classes, w):
                if int(cid) < 2:
                    w_full[int(cid)] = float(ww)
            self.class_weights = torch.tensor(w_full, dtype=torch.float32)

    def loaders(self) -> Tuple[DataLoader, DataLoader]:
        if self.X_train is None:
            raise RuntimeError("Call prepare() first")

        X_train_t = torch.tensor(self.X_train, dtype=torch.float32)
        y_train_t = torch.tensor(self.y_train, dtype=torch.long)
        X_test_t = torch.tensor(self.X_test, dtype=torch.float32)
        y_test_t = torch.tensor(self.y_test, dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory
        )
        test_loader = DataLoader(
            TensorDataset(X_test_t, y_test_t),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory
        )
        return train_loader, test_loader

    def summary(self) -> None:
        print("CICIoT23 prepared ✅ (binary)")
        print("scale:", self.cfg.scale)
        print("X_train:", None if self.X_train is None else self.X_train.shape)
        print("X_test :", None if self.X_test is None else self.X_test.shape)
        if self.y_test is not None:
            u, c = np.unique(self.y_test, return_counts=True)
            print("y_test counts:", dict(zip(u.tolist(), c.tolist())))
        if self.class_weights is not None:
            print("Class weights:", self.class_weights.tolist())
