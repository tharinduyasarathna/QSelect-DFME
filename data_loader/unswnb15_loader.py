# ============================================================
# FILE: data_loader/unswnb15_loader.py


import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import TensorDataset, DataLoader

from .base_tabular import BaseTabularConfig, BaseTabularDataModule


# -----------------------------
# Default columns to drop
# -----------------------------
DEFAULT_DROP_COLS = [
    "id",           # row id in many UNSW exports
    "attack_cat",   # multiclass category (drop for binary)
]


@dataclass
class UNSWNB15Config(BaseTabularConfig):
    csv_dir: str = ""
    train_file: str = "UNSW_NB15_training-set.csv"
    test_file: str = "UNSW_NB15_testing-set.csv"

    label_col: str = "label"     # official is 'label' (0/1)
    drop_cols: Optional[List[str]] = None

    # binary only (kept for consistency)
    task: str = "binary"         # only supported: "binary"

    # max_rows applies to TRAIN ONLY (keeps official test intact)
    # (BaseTabularConfig already has max_rows)


class UNSWNB15DataModule(BaseTabularDataModule):
    """
    UNSW-NB15 binary loader.
    Uses official train/test CSVs when available.
    """

    def __init__(self, cfg: UNSWNB15Config):
        super().__init__(cfg)
        self.cfg: UNSWNB15Config = cfg
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
        return "UNSW-NB15"

    # -----------------------------
    # IO helpers
    # -----------------------------
    def _resolve_paths(self) -> Tuple[str, str]:
        train_path = os.path.join(self.cfg.csv_dir, self.cfg.train_file)
        test_path = os.path.join(self.cfg.csv_dir, self.cfg.test_file)

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Missing train CSV: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing test CSV: {test_path}")

        return train_path, test_path

    def _load_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tr_path, te_path = self._resolve_paths()
        train_df = pd.read_csv(tr_path, low_memory=False)
        test_df = pd.read_csv(te_path, low_memory=False)
        return train_df, test_df

    # -----------------------------
    # Stratified subsample (TRAIN only)
    # -----------------------------
    def _stratified_subsample_train(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
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
    # Cleaning / encoding
    # -----------------------------
    def _clean_common(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        # label column name robustness
        if self.cfg.label_col not in df.columns:
            # sometimes "Label" exists in other repos
            if "Label" in df.columns:
                df = df.rename(columns={"Label": self.cfg.label_col})
            else:
                raise KeyError(f"Label column '{self.cfg.label_col}' not found.")

        # drop selected columns
        drop_cols = self.cfg.drop_cols if self.cfg.drop_cols is not None else DEFAULT_DROP_COLS
        drop_cols = [c for c in drop_cols if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # replace inf/-inf -> NaN
        df = df.replace([np.inf, -np.inf, "Infinity", "inf", "-inf"], np.nan)

        # ensure label is numeric 0/1
        df[self.cfg.label_col] = pd.to_numeric(df[self.cfg.label_col], errors="coerce")
        df = df.dropna(subset=[self.cfg.label_col])
        df[self.cfg.label_col] = df[self.cfg.label_col].astype(int)
        df = df[df[self.cfg.label_col].isin([0, 1])].copy()

        return df

    def _one_hot_fit_transform(self, X_train_df: pd.DataFrame, X_test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        One-hot encode object/category columns using train schema,
        then align test to train columns.
        """
        # Identify categorical columns
        cat_cols = X_train_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # Convert bool -> string to avoid weirdness
        for c in cat_cols:
            X_train_df[c] = X_train_df[c].astype(str)
            X_test_df[c] = X_test_df[c].astype(str)

        # get_dummies on train, then reindex test
        X_train_oh = pd.get_dummies(X_train_df, columns=cat_cols, drop_first=False)
        X_test_oh = pd.get_dummies(X_test_df, columns=cat_cols, drop_first=False)

        X_test_oh = X_test_oh.reindex(columns=X_train_oh.columns, fill_value=0)

        return (
            X_train_oh.to_numpy(dtype=np.float32),
            X_test_oh.to_numpy(dtype=np.float32),
        )

    def _impute_and_sanitize(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # convert any leftover non-finite to NaN, then fill with train medians
        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)

        X_train = np.where(np.isfinite(X_train), X_train, np.nan).astype(np.float32)
        X_test = np.where(np.isfinite(X_test), X_test, np.nan).astype(np.float32)

        # median impute using train only
        med = np.nanmedian(X_train, axis=0)
        # if a whole column is NaN, nanmedian gives NaN -> replace with 0
        med = np.nan_to_num(med, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        inds_tr = np.isnan(X_train)
        if inds_tr.any():
            X_train[inds_tr] = np.take(med, np.where(inds_tr)[1])

        inds_te = np.isnan(X_test)
        if inds_te.any():
            X_test[inds_te] = np.take(med, np.where(inds_te)[1])

        # final hard safety
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return X_train, X_test

    # -----------------------------
    # Main pipeline
    # -----------------------------
    def prepare(self) -> None:
        train_df, test_df = self._load_raw()

        train_df = self._clean_common(train_df)
        test_df = self._clean_common(test_df)

        # optional: limit TRAIN size only (keep official test intact)
        if self.cfg.max_rows is not None:
            train_df = self._stratified_subsample_train(train_df, int(self.cfg.max_rows))

        y_train = train_df[self.cfg.label_col].to_numpy(dtype=np.int64)
        y_test = test_df[self.cfg.label_col].to_numpy(dtype=np.int64)

        X_train_df = train_df.drop(columns=[self.cfg.label_col])
        X_test_df = test_df.drop(columns=[self.cfg.label_col])

        # coerce numeric columns where possible (keeps objects for one-hot)
        for c in X_train_df.columns:
            if c not in X_train_df.select_dtypes(include=["object", "category", "bool"]).columns:
                X_train_df[c] = pd.to_numeric(X_train_df[c], errors="coerce")
        for c in X_test_df.columns:
            if c not in X_test_df.select_dtypes(include=["object", "category", "bool"]).columns:
                X_test_df[c] = pd.to_numeric(X_test_df[c], errors="coerce")

        # one-hot (train schema)
        X_train, X_test = self._one_hot_fit_transform(X_train_df, X_test_df)

        # impute + sanitize
        X_train, X_test = self._impute_and_sanitize(X_train, X_test)

        # rebuild qt_minmax with safe n_quantiles based on train size
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

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        # class weights (balanced) for CrossEntropy/BCE usage
        classes = np.unique(self.y_train)
        if len(classes) >= 2:
            w = compute_class_weight(class_weight="balanced", classes=classes, y=self.y_train)
            w_full = np.zeros((2,), dtype=np.float32)
            for cid, ww in zip(classes, w):
                if int(cid) in (0, 1):
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
        print("UNSW-NB15 prepared ✅ (binary)")
        print("scale:", self.cfg.scale)
        print("X_train:", None if self.X_train is None else self.X_train.shape)
        print("X_test :", None if self.X_test is None else self.X_test.shape)
        print("Label map:", self.label_to_id)
        if self.class_weights is not None:
            print("Class weights:", self.class_weights.tolist())
        if self.y_test is not None:
            u, c = np.unique(self.y_test, return_counts=True)
            print("y_test counts:", dict(zip(u.tolist(), c.tolist())))
