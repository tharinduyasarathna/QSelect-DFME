# data_loader/base_tabular.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import TensorDataset, DataLoader


@dataclass
class BaseTabularConfig:
    """
    Generic tabular config used by UCI/OpenML loaders.
    """
    label_col: str = "target"

    test_size: float = 0.2
    random_state: int = 42

    batch_size: int = 512
    num_workers: int = 0
    pin_memory: bool = True

    max_rows: Optional[int] = None  # optional subsample for fast dev
    drop_cols: Optional[List[str]] = None  # optional columns to drop

    # scaling modes:
    #   - "zscore"    : StandardScaler
    #   - "qt_minmax" : QuantileTransformer(normal) + MinMaxScaler(0,1)  ✅ for DFME + sigmoid generator
    #   - "none"      : no scaling
    scale: str = "zscore"


class BaseTabularDataModule:
    """
    Generic tabular DataModule:
      - Load raw df (subclass)
      - Clean + basic preprocessing (base)
      - Encode binary labels (subclass or base default)
      - Train/test split (stratified)
      - Scale numeric (train-fit only)
      - DataLoaders
    """

    def __init__(self, cfg: BaseTabularConfig):
        self.cfg = cfg
        self.scaler = self._build_scaler(cfg)

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        self.class_weights: Optional[torch.Tensor] = None
        self.meta: Dict[str, str] = {}

    # -----------------------------
    # scaler builder
    # -----------------------------
    def _build_scaler(self, cfg: BaseTabularConfig):
        scale = str(cfg.scale).lower().strip()

        if scale == "none":
            return None

        if scale == "zscore":
            return StandardScaler(with_mean=True, with_std=True)
        
        if scale == "minmax":
            return MinMaxScaler(feature_range=(0.0, 1.0))

        if scale in ("qt_minmax", "qt+minmax", "minmax_01"):
            # placeholder pipeline; rebuilt safely after split in prepare()
            qt = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=1000,
                subsample=200_000,
                random_state=cfg.random_state
            )
            mm = MinMaxScaler(feature_range=(0.0, 1.0))
            return Pipeline([("qt", qt), ("mm", mm)])

        raise ValueError(f"Unknown scale='{cfg.scale}'. Use: zscore | qt_minmax | none")

    # -----------------------------
    # to be implemented by subclasses
    # -----------------------------
    def _load_raw_df(self) -> pd.DataFrame:
        raise NotImplementedError

    # -----------------------------
    # label mapping (override if you need dataset-specific)
    # -----------------------------
    def _binary_map(self, y: pd.Series) -> np.ndarray:
        """
        Default rule:
          - If labels are already {0,1}, keep them
          - Else: map most frequent label -> 0 (normal), others -> 1 (anomaly)
        """
        y = y.astype(str)
        uniq = set(y.unique().tolist())
        if uniq <= {"0", "1"}:
            return y.astype(int).to_numpy(dtype=np.int64)

        vc = y.value_counts()
        normal_label = vc.index[0]
        return (y != normal_label).astype(int).to_numpy(dtype=np.int64)

    # -----------------------------
    # base cleaning + encoding
    # -----------------------------
    def _clean_and_filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # normalize column names
        df.columns = [c.strip() for c in df.columns]

        if self.cfg.label_col not in df.columns:
            raise KeyError(
                f"Label column '{self.cfg.label_col}' not found in df columns={df.columns.tolist()[:20]}..."
            )

        # drop configured cols if present
        if self.cfg.drop_cols:
            drop_cols = [c for c in self.cfg.drop_cols if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)

        # optional subsample (deterministic)
        if self.cfg.max_rows is not None and len(df) > self.cfg.max_rows:
            df = df.sample(n=int(self.cfg.max_rows), random_state=self.cfg.random_state).reset_index(drop=True)

        # replace inf strings and inf values
        df = df.replace([np.inf, -np.inf, "Infinity", "inf", "-inf"], np.nan)

        # separate y
        y = df[self.cfg.label_col]
        X = df.drop(columns=[self.cfg.label_col])

        # coerce numeric where possible, keep categoricals
        for c in X.columns:
            if X[c].dtype == object:
                tmp = pd.to_numeric(X[c], errors="coerce")
                na_rate = float(tmp.isna().mean())
                if na_rate < 0.30:
                    X[c] = tmp
                else:
                    X[c] = X[c].astype(str).str.strip()

        # fill numeric NaNs with median
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            med = X[num_cols].median(numeric_only=True)
            X[num_cols] = X[num_cols].fillna(med)

        # fill categorical NaNs with "missing"
        cat_cols = [c for c in X.columns if c not in num_cols]
        if cat_cols:
            for c in cat_cols:
                X[c] = X[c].astype(object)
            X[cat_cols] = X[cat_cols].fillna("missing")
            X[cat_cols] = X[cat_cols].astype(str)

        # one-hot encode categoricals
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

        # drop constant columns
        nunique = X.nunique(dropna=False)
        const_cols = nunique[nunique <= 1].index.tolist()
        if const_cols:
            X = X.drop(columns=const_cols)

        out = X.copy()
        out[self.cfg.label_col] = y.astype(str).values
        out = out.dropna()
        return out

    # -----------------------------
    # main pipeline
    # -----------------------------
    def prepare(self) -> None:
        df = self._load_raw_df()
        df = self._clean_and_filter_df(df)

        # extract arrays
        y_raw = df[self.cfg.label_col]
        y = self._binary_map(y_raw)  # 0/1
        X_df = df.drop(columns=[self.cfg.label_col])
        X = X_df.to_numpy(dtype=np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y
        )

        # If using qt_minmax, rebuild scaler with safe n_quantiles based on train size
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

        # class weights (for optional supervised baselines)
        classes = np.unique(self.y_train)
        if len(classes) >= 2:
            w = compute_class_weight(class_weight="balanced", classes=classes, y=self.y_train)
            w_full = np.zeros((2,), dtype=np.float32)
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
        print("Tabular dataset prepared ✅")
        print("scale:", self.cfg.scale)
        print("X_train:", None if self.X_train is None else self.X_train.shape)
        print("X_test :", None if self.X_test is None else self.X_test.shape)
        if self.y_test is not None:
            u, c = np.unique(self.y_test, return_counts=True)
            print("y_test counts:", dict(zip(u.tolist(), c.tolist())))
        if self.class_weights is not None:
            print("Class weights:", self.class_weights.tolist())



