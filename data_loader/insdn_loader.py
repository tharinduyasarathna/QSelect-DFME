# ============================================================
# FILE: data_loader/insdn_loader.py  

import os
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


DEFAULT_DROP_COLS = [
    "Flow ID",
    "Src IP", "Source IP", "SourceIP",
    "Dst IP", "Destination IP", "DestinationIP",
    "Timestamp", "TimeStamp", "time", "Time",
    "Src Port", "Source Port", "Sport",
    "Dst Port", "Destination Port", "Dport",
    "Fwd Header Length.1",
]

DEFAULT_LABEL_MAP_5 = {
    "NORMAL": "BENIGN",
    "BENIGN": "BENIGN",
    "DOS": "DoS",
    "DDOS": "DDoS",
    "PROBE": "Probe",
    "BOTNET": "Botnet",
    "BOT": "Botnet",
    "DDOS ": "DDoS",
}
DEFAULT_KEEP_LABELS_5 = ("BENIGN", "DoS", "DDoS", "Probe", "Botnet")
DEFAULT_MULTICLASS_ORDER_5 = ("BENIGN", "DoS", "DDoS", "Probe", "Botnet")


def _norm_label(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip().str.upper()
    out = out.str.replace(r"\s+", " ", regex=True)
    return out


@dataclass
class InSDNConfig(BaseTabularConfig):
    csv_dir: str = ""
    files: Optional[Sequence[str]] = None
    label_col: str = "Label"

    task: str = "multiclass"  # "multiclass" | "binary"

    # default scaling preference
    scale: str = "minmax"

    # columns
    drop_cols: Optional[List[str]] = None

    # OPTIONAL: enforce schema if you want
    keep_only_cols: bool = False
    keep_cols: Optional[List[str]] = None

    # label mapping
    label_map: Optional[Dict[str, str]] = None
    keep_labels: Optional[Sequence[str]] = None
    multiclass_order: Optional[Sequence[str]] = None

    # robustness
    drop_duplicates: bool = True
    drop_constant_cols: bool = True
    drop_nan_frac: float = 0.95

    protocol_col: str = "Protocol"
    onehot_protocol: bool = False

    # DFME stability
    clip_raw: Optional[float] = None
    clip_after_scale: Optional[float] = None


class InSDNDataModule(BaseTabularDataModule):
    def __init__(self, cfg: InSDNConfig):
        super().__init__(cfg)
        self.cfg: InSDNConfig = cfg
        self.label_to_id: Dict[str, int] = {}

        scale = str(cfg.scale).lower().strip()
        if scale == "zscore":
            self.scaler = StandardScaler(with_mean=True, with_std=True)
        elif scale == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
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
            raise ValueError(f"Unknown scale='{cfg.scale}'. Use: zscore | minmax | qt_minmax | none")

    def dataset_name(self) -> str:
        return "InSDN"

    def _resolve_files(self) -> List[str]:
        if self.cfg.files is None:
            cand = ["Normal_data.csv", "OVS.csv", "metasploitable-2.csv"]
        else:
            cand = list(self.cfg.files)

        files = []
        for name in cand:
            path = os.path.join(self.cfg.csv_dir, name)
            if os.path.exists(path):
                files.append(path)

        if not files:
            raise FileNotFoundError(
                f"No InSDN CSV files found in: {self.cfg.csv_dir}\n"
                f"Tried: {cand}\n"
                f"Set InSDNConfig(files=...) if your filenames differ."
            )
        return files

    def _load_raw_df(self) -> pd.DataFrame:
        dfs = []
        for p in self._resolve_files():
            df = pd.read_csv(p, low_memory=False)
            dfs.append(df)
        return pd.concat(dfs, axis=0, ignore_index=True)

    def _stratified_subsample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if n is None or len(df) <= n:
            return df

        rng = np.random.RandomState(self.cfg.random_state)
        y = df[self.cfg.label_col].astype(str)
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

    def _enforce_keep_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        if not bool(self.cfg.keep_only_cols):
            return df
        if not self.cfg.keep_cols:
            raise ValueError("keep_only_cols=True but keep_cols is None/empty.")
        keep = [c.strip() for c in self.cfg.keep_cols]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required keep_cols columns: {missing}")
        return df.loc[:, keep].copy()

    def _clean_and_filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        if self.cfg.label_col not in df.columns:
            raise KeyError(f"Label column '{self.cfg.label_col}' not found in CSV.")

        # drop leakage columns
        drop_cols = self.cfg.drop_cols if self.cfg.drop_cols is not None else DEFAULT_DROP_COLS
        drop_cols = [c for c in drop_cols if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # optional schema enforcement
        df = self._enforce_keep_cols(df)

        # duplicates help stability
        if bool(self.cfg.drop_duplicates):
            df = df.drop_duplicates(keep="first")

        # normalize labels -> map -> keep only known classes
        raw_u = _norm_label(df[self.cfg.label_col])

        label_map = self.cfg.label_map if self.cfg.label_map is not None else DEFAULT_LABEL_MAP_5
        label_map_u = {str(k).strip().upper(): v for k, v in label_map.items()}
        df[self.cfg.label_col] = raw_u.map(label_map_u)

        keep = tuple(self.cfg.keep_labels) if self.cfg.keep_labels is not None else DEFAULT_KEEP_LABELS_5
        df = df[df[self.cfg.label_col].isin(set(keep))].copy()

        # max_rows AFTER filtering
        if self.cfg.max_rows is not None:
            df = self._stratified_subsample(df, int(self.cfg.max_rows))

        # replace inf
        df = df.replace([np.inf, -np.inf, "Infinity", "inf", "-inf"], np.nan)

        # protocol handling
        if self.cfg.protocol_col in df.columns:
            if bool(self.cfg.onehot_protocol):
                prot = pd.to_numeric(df[self.cfg.protocol_col], errors="coerce").fillna(-1).astype(int)
                dummies = pd.get_dummies(prot, prefix=self.cfg.protocol_col)
                df = pd.concat([df.drop(columns=[self.cfg.protocol_col]), dummies], axis=1)
            else:
                df[self.cfg.protocol_col] = pd.to_numeric(df[self.cfg.protocol_col], errors="coerce")

        # numeric conversion for all non-label columns
        feat_cols = [c for c in df.columns if c != self.cfg.label_col]
        for c in feat_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # drop mostly-NaN columns
        nan_frac = df[feat_cols].isna().mean(axis=0)
        bad_cols = nan_frac[nan_frac >= float(self.cfg.drop_nan_frac)].index.tolist()
        if bad_cols:
            df = df.drop(columns=bad_cols)
            feat_cols = [c for c in df.columns if c != self.cfg.label_col]

        # fill
        med = df[feat_cols].median(numeric_only=True)
        df[feat_cols] = df[feat_cols].fillna(med)
        df[feat_cols] = df[feat_cols].fillna(0.0)

        # drop constant features
        if bool(self.cfg.drop_constant_cols):
            nunique = df[feat_cols].nunique(dropna=False)
            const_cols = nunique[nunique <= 1].index.tolist()
            if const_cols:
                df = df.drop(columns=const_cols)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df

    def _encode_labels_multiclass(self, y: pd.Series) -> np.ndarray:
        order = tuple(self.cfg.multiclass_order) if self.cfg.multiclass_order is not None else DEFAULT_MULTICLASS_ORDER_5
        present = [c for c in order if c in set(y.astype(str).unique())]
        self.label_to_id = {lab: i for i, lab in enumerate(present)}
        return y.astype(str).map(self.label_to_id).to_numpy(dtype=np.int64)

    def _encode_labels_binary(self, y: pd.Series) -> np.ndarray:
        ys = y.astype(str)
        return (ys != "BENIGN").astype(int).to_numpy(dtype=np.int64)

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
            n_classes = max(2, len(self.label_to_id))

        X = X_df.to_numpy(dtype=np.float32, copy=False)

        # raw clip optional
        if self.cfg.clip_raw is not None and float(self.cfg.clip_raw) > 0:
            X = np.clip(X, -float(self.cfg.clip_raw), float(self.cfg.clip_raw))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y
        )

        # rebuild qt_minmax with safe n_quantiles
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
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        X_train = X_train.astype(np.float32, copy=False)
        X_test = X_test.astype(np.float32, copy=False)

        if self.cfg.clip_after_scale is not None and float(self.cfg.clip_after_scale) > 0:
            X_train = np.clip(X_train, -float(self.cfg.clip_after_scale), float(self.cfg.clip_after_scale))
            X_test = np.clip(X_test, -float(self.cfg.clip_after_scale), float(self.cfg.clip_after_scale))

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        self.X_train, self.y_train = X_train, y_train.astype(np.int64)
        self.X_test, self.y_test = X_test, y_test.astype(np.int64)

        classes = np.unique(self.y_train)
        if len(classes) >= 2:
            w = compute_class_weight(class_weight="balanced", classes=classes, y=self.y_train)
            w_full = np.zeros((n_classes,), dtype=np.float32)
            for cid, ww in zip(classes, w):
                if int(cid) < n_classes:
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
        print(f"InSDN prepared ✅ ({task})")
        print("scale:", self.cfg.scale)
        print("onehot_protocol:", bool(self.cfg.onehot_protocol))
        print("keep_only_cols:", bool(self.cfg.keep_only_cols))
        print("X_train:", None if self.X_train is None else self.X_train.shape)
        print("X_test :", None if self.X_test is None else self.X_test.shape)
        print("Label map:", self.label_to_id)
        if self.class_weights is not None:
            print("Class weights:", self.class_weights.tolist())
        if self.y_test is not None:
            u, c = np.unique(self.y_test, return_counts=True)
            print("y_test counts:", dict(zip(u.tolist(), c.tolist())))
        if self.X_train is not None and self.X_test is not None:
            nbad_tr = int(np.sum(~np.isfinite(self.X_train)))
            nbad_te = int(np.sum(~np.isfinite(self.X_test)))
            print("non-finite entries (train/test):", nbad_tr, nbad_te)