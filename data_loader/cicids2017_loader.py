# data_loader/cicids2017_loader.py
import os
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import TensorDataset, DataLoader

from .base_tabular import BaseTabularConfig, BaseTabularDataModule


# -----------------------------
# CICIDS2017 label normalization to 4 classes
# -----------------------------
LABEL_MAP = {
    "BENIGN": "BENIGN",

    # DoS variants
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS slowloris": "DoS",

    # DDoS
    "DDoS": "DDoS",

    # Botnet
    "Bot": "BOTNET",
}
KEEP_LABELS = {"BENIGN", "DoS", "DDoS", "BOTNET"}


# Leakage / identifier columns to drop if present
DEFAULT_DROP_COLS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Fwd Header Length.1",
]


@dataclass
class CICIDSConfig(BaseTabularConfig):
    csv_dir: str = ""
    label_col: str = "Label"
    drop_cols: Optional[List[str]] = None

    # choose label mode
    task: str = "multiclass"   # "multiclass" | "binary"


class CICIDS2017DataModule(BaseTabularDataModule):
    """
    CICIDS2017 loader supporting BOTH:
      - multiclass: {BENIGN, DoS, DDoS, BOTNET} -> {0..3}
      - binary: BENIGN -> 0, (DoS/DDoS/BOTNET) -> 1

    Key fix:
      max_rows is applied AFTER label mapping + filtering, via stratified sampling.
      This prevents the "only BENIGN+DDoS" problem when taking first N rows.
    """

    def __init__(self, cfg: CICIDSConfig):
        super().__init__(cfg)
        self.cfg: CICIDSConfig = cfg
        self.label_to_id: Dict[str, int] = {}

        # IMPORTANT:
        # - DO NOT override BaseTabularDataModule scaling logic in a way that breaks qt_minmax.
        # - If cfg.scale == "zscore", keep StandardScaler.
        # - If cfg.scale == "qt_minmax", build pipeline (rebuilt safely later in prepare()).
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
        return "CICIDS2017"

    # -----------------------------
    # IO
    # -----------------------------
    def _load_raw_df(self) -> pd.DataFrame:
        pattern = os.path.join(self.cfg.csv_dir, "*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No .csv files found in: {self.cfg.csv_dir}\n"
                f"Tip: make sure the directory contains CICIDS2017 CSVs."
            )

        dfs = []
        for f in files:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)

        return pd.concat(dfs, axis=0, ignore_index=True)

    # -----------------------------
    # Stratified subsampling AFTER filtering
    # -----------------------------
    def _stratified_subsample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if n is None or len(df) <= n:
            return df

        rng = np.random.RandomState(self.cfg.random_state)
        y = df[self.cfg.label_col].astype(str)
        vc = y.value_counts()
        labels = vc.index.tolist()
        counts = vc.values.astype(int)

        # proportional allocation
        alloc = np.floor(counts / counts.sum() * n).astype(int)
        alloc = np.maximum(alloc, 1)  # ensure at least 1 per class

        # adjust rounding to hit exactly n
        diff = int(n) - int(alloc.sum())
        if diff > 0:
            order = np.argsort(-counts)  # give extras to largest classes
            for i in range(diff):
                alloc[order[i % len(order)]] += 1
        elif diff < 0:
            order = np.argsort(-alloc)   # remove from largest allocations
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
    # Cleaning / filtering
    # -----------------------------
    def _clean_and_filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        if self.cfg.label_col not in df.columns:
            raise KeyError(f"Label column '{self.cfg.label_col}' not found.")

        # drop leakage columns
        drop_cols = self.cfg.drop_cols if self.cfg.drop_cols is not None else DEFAULT_DROP_COLS
        drop_cols = [c for c in drop_cols if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # normalize labels
        df[self.cfg.label_col] = df[self.cfg.label_col].astype(str).str.strip()
        df[self.cfg.label_col] = df[self.cfg.label_col].map(LABEL_MAP)

        # keep only 4 classes
        df = df[df[self.cfg.label_col].isin(KEEP_LABELS)].copy()

        # IMPORTANT: apply max_rows AFTER label filtering so we keep all 4 classes
        if self.cfg.max_rows is not None:
            df = self._stratified_subsample(df, int(self.cfg.max_rows))

        # replace infs and strings
        df = df.replace([np.inf, -np.inf, "Infinity", "inf", "-inf"], np.nan)

        # coerce features numeric
        feat_cols = [c for c in df.columns if c != self.cfg.label_col]
        for c in feat_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # drop missing labels
        df = df.dropna(subset=[self.cfg.label_col])

        # fill NaNs in features with median
        med = df[feat_cols].median(numeric_only=True)
        df[feat_cols] = df[feat_cols].fillna(med)

        # drop constant columns
        nunique = df[feat_cols].nunique(dropna=False)
        const_cols = nunique[nunique <= 1].index.tolist()
        if const_cols:
            df = df.drop(columns=const_cols)

        # final safety
        df = df.dropna()
        return df

    # -----------------------------
    # Label encoding
    # -----------------------------
    def _encode_labels_multiclass(self, y: pd.Series) -> np.ndarray:
        order = ["BENIGN", "DoS", "DDoS", "BOTNET"]
        present = [c for c in order if c in set(y.astype(str).unique())]
        self.label_to_id = {lab: i for i, lab in enumerate(present)}
        return y.astype(str).map(self.label_to_id).to_numpy(dtype=np.int64)

    def _encode_labels_binary(self, y: pd.Series) -> np.ndarray:
        ys = y.astype(str)
        return (ys != "BENIGN").astype(int).to_numpy(dtype=np.int64)

    # -----------------------------
    # Main pipeline
    # -----------------------------
    def prepare(self) -> None:
        df = self._load_raw_df()
        df = self._clean_and_filter_df(df)

        y_raw = df[self.cfg.label_col]
        X_df = df.drop(columns=[self.cfg.label_col])

        task = str(self.cfg.task).lower().strip()
        if task == "binary":
            y = self._encode_labels_binary(y_raw)
            n_classes = 2
            self.label_to_id = {"BENIGN": 0, "ATTACK": 1}
        else:
            y = self._encode_labels_multiclass(y_raw)
            n_classes = len(self.label_to_id)

        X = X_df.to_numpy(dtype=np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y
        )

        # If using qt_minmax, rebuild with safe n_quantiles based on train size
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

        # scale (fit on train only)
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train).astype(np.float32)
            X_test = self.scaler.transform(X_test).astype(np.float32)

        self.X_train, self.y_train = X_train, y_train.astype(np.int64)
        self.X_test, self.y_test = X_test, y_test.astype(np.int64)

        # class weights (balanced) for CrossEntropy
        classes = np.unique(self.y_train)
        if len(classes) >= 2:
            w = compute_class_weight(class_weight="balanced", classes=classes, y=self.y_train)
            w_full = np.zeros((n_classes,), dtype=np.float32)
            for cid, ww in zip(classes, w):
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
        task = str(self.cfg.task).lower().strip()
        print(f"CICIDS2017 prepared ✅ ({task})")
        print("scale:", self.cfg.scale)
        print("X_train:", None if self.X_train is None else self.X_train.shape)
        print("X_test :", None if self.X_test is None else self.X_test.shape)
        print("Label map:", self.label_to_id)
        if self.class_weights is not None:
            print("Class weights:", self.class_weights.tolist())
        if self.y_test is not None:
            u, c = np.unique(self.y_test, return_counts=True)
            print("y_test counts:", dict(zip(u.tolist(), c.tolist())))


