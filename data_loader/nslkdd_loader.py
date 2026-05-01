# ============================================================
# FILE: data_loader/nslkdd_loader.py
# ============================================================
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
# NSL-KDD attack grouping (5-class)
# -----------------------------
# Based on the common KDD/NSL-KDD taxonomy:
# NORMAL vs {DoS, Probe, R2L, U2R}
DOS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "mailbomb", "processtable", "udpstorm", "apache2", "worm"
}
PROBE = {
    "ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"
}
R2L = {
    "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
    "warezclient", "warezmaster", "xlock", "xsnoop", "snmpguess",
    "snmpgetattack", "httptunnel", "sendmail", "named"
}
U2R = {
    "buffer_overflow", "loadmodule", "perl", "rootkit", "sqlattack", "xterm", "ps"
}

# If your CSV has other/unknown attack labels, we will map them to "ATTACK_OTHER"
# (and in multiclass mode they’ll become R2L by default? No — we keep a separate bucket optionally.)
# To keep the exact 5-class setup, we map unknown attacks to PROBE (conservative) or ATTACK_OTHER.
# Here: map unknown to "ATTACK_OTHER" and then drop it into PROBE bucket unless you want strict-only.
UNKNOWN_TO = "ATTACK_OTHER"


DEFAULT_CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
DEFAULT_DROP_COLS = ["difficulty"]  # many NSL-KDD exports include this


@dataclass
class NSLKDDConfig(BaseTabularConfig):
    csv_dir: str = ""
    label_col: str = "label"          # many CSVs use: label / class / attack
    drop_cols: Optional[List[str]] = None

    # choose label mode
    task: str = "binary"              # "binary" | "multiclass"

    # if your CSV has no header, set this True (auto-detect also works)
    header: Optional[bool] = None     # None=auto, True=has header, False=no header


class NSLKDDDataModule(BaseTabularDataModule):
    """
    NSL-KDD loader (CSV) supporting BOTH:
      - binary: NORMAL -> 0, ATTACK -> 1
      - multiclass: {NORMAL, DoS, PROBE, R2L, U2R} -> {0..4}

    Key fix (same as CICIDS loader):
      max_rows is applied AFTER label mapping + filtering, via stratified sampling.
    """

    def __init__(self, cfg: NSLKDDConfig):
        super().__init__(cfg)
        self.cfg: NSLKDDConfig = cfg
        self.label_to_id: Dict[str, int] = {}

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
        elif scale == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        elif scale == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown scale='{cfg.scale}'. Use: zscore | qt_minmax | minmax | none")

    def dataset_name(self) -> str:
        return "NSL-KDD"

    # -----------------------------
    # IO
    # -----------------------------
    def _load_raw_df(self) -> pd.DataFrame:
        pattern_csv = os.path.join(self.cfg.csv_dir, "*.csv")
        files = sorted(glob.glob(pattern_csv))
        if not files:
            raise FileNotFoundError(
                f"No .csv files found in: {self.cfg.csv_dir}\n"
                f"Tip: put NSL-KDD CSVs inside that directory."
            )

        dfs = []
        for f in files:
            # header auto-detection:
            # - If cfg.header is True/False, enforce it.
            # - If None, read once with header=0 and check if label_col appears; otherwise read header=None.
            if self.cfg.header is True:
                df = pd.read_csv(f, low_memory=False)
            elif self.cfg.header is False:
                df = pd.read_csv(f, header=None, low_memory=False)
            else:
                df_try = pd.read_csv(f, low_memory=False)
                cols_lower = set([str(c).strip().lower() for c in df_try.columns])
                if str(self.cfg.label_col).strip().lower() in cols_lower:
                    df = df_try
                else:
                    # fallback to no-header
                    df = pd.read_csv(f, header=None, low_memory=False)

            dfs.append(df)

        df_all = pd.concat(dfs, axis=0, ignore_index=True)
        return df_all

    # -----------------------------
    # Stratified subsampling AFTER filtering
    # -----------------------------
    def _stratified_subsample(self, df: pd.DataFrame, y_col: str, n: int) -> pd.DataFrame:
        if n is None or len(df) <= int(n):
            return df

        rng = np.random.RandomState(self.cfg.random_state)
        y = df[y_col].astype(str)
        vc = y.value_counts()
        labels = vc.index.tolist()
        counts = vc.values.astype(int)

        alloc = np.floor(counts / counts.sum() * int(n)).astype(int)
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
            sub = df[df[y_col] == lab]
            if len(sub) <= int(take):
                parts.append(sub)
            else:
                idx = rng.choice(len(sub), size=int(take), replace=False)
                parts.append(sub.iloc[idx])

        out = pd.concat(parts, axis=0, ignore_index=True)
        out = out.sample(frac=1.0, random_state=self.cfg.random_state).reset_index(drop=True)
        return out

    # -----------------------------
    # Helpers: find label column robustly
    # -----------------------------
    def _infer_label_col(self, df: pd.DataFrame) -> str:
        # If headerless, df columns are integers -> label is last column by convention.
        if all(isinstance(c, (int, np.integer)) for c in df.columns):
            return df.columns[-1]

        cols = [str(c).strip() for c in df.columns]
        cols_lower = [c.lower() for c in cols]

        # preferred
        want = str(self.cfg.label_col).strip().lower()
        if want in cols_lower:
            return cols[cols_lower.index(want)]

        # common alternatives
        for cand in ["label", "class", "attack", "target", "y"]:
            if cand in cols_lower:
                return cols[cols_lower.index(cand)]

        # fallback: last column
        return df.columns[-1]

    def _normalize_attack_label(self, s: pd.Series) -> pd.Series:
        # Many exports have trailing '.' like "neptune." or "normal."
        out = s.astype(str).str.strip()
        out = out.str.replace(r"\.+$", "", regex=True)
        out = out.str.lower()
        return out

    def _map_to_5class(self, raw_attack: pd.Series) -> pd.Series:
        a = self._normalize_attack_label(raw_attack)

        def map_one(x: str) -> str:
            if x in ("normal", "benign"):
                return "NORMAL"
            if x in DOS:
                return "DoS"
            if x in PROBE:
                return "PROBE"
            if x in R2L:
                return "R2L"
            if x in U2R:
                return "U2R"
            return UNKNOWN_TO

        return a.map(map_one)

    # -----------------------------
    # Main cleaning / encoding
    # -----------------------------
    def _clean_and_filter_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        df = df.copy()

        # If headerless, create generic names
        if all(isinstance(c, (int, np.integer)) for c in df.columns):
            df.columns = [f"f{i}" for i in range(df.shape[1])]

        df.columns = [str(c).strip() for c in df.columns]
        label_col = self._infer_label_col(df)

        # drop configured columns
        drop_cols = self.cfg.drop_cols if self.cfg.drop_cols is not None else DEFAULT_DROP_COLS
        drop_cols = [c for c in drop_cols if c in df.columns and c != label_col]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # label normalize -> attack group
        df[label_col] = df[label_col].astype(str)
        df["_group_label"] = self._map_to_5class(df[label_col])

        # optional: if you want strict 5-class only, drop ATTACK_OTHER rows:
        # df = df[df["_group_label"] != "ATTACK_OTHER"].copy()

        # apply max_rows AFTER mapping (same as CICIDS fix)
        if self.cfg.max_rows is not None:
            df = self._stratified_subsample(df, y_col="_group_label", n=int(self.cfg.max_rows))

        # replace infs
        df = df.replace([np.inf, -np.inf, "Infinity", "inf", "-inf"], np.nan)

        # identify categorical columns if present
        cat_cols = [c for c in DEFAULT_CATEGORICAL_COLS if c in df.columns and c != label_col]
        feat_cols = [c for c in df.columns if c not in (label_col, "_group_label")]

        # coerce numeric for non-categorical features
        for c in feat_cols:
            if c in cat_cols:
                df[c] = df[c].astype(str).fillna("UNK")
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # fill NaNs in numeric features
        num_cols = [c for c in feat_cols if c not in cat_cols]
        if num_cols:
            med = df[num_cols].median(numeric_only=True)
            df[num_cols] = df[num_cols].fillna(med)

        # fill NaNs in categorical
        for c in cat_cols:
            df[c] = df[c].fillna("UNK")

        # drop constant numeric columns
        if num_cols:
            nunique = df[num_cols].nunique(dropna=False)
            const_cols = nunique[nunique <= 1].index.tolist()
            if const_cols:
                df = df.drop(columns=const_cols)

        df = df.dropna(subset=["_group_label"]).reset_index(drop=True)
        return df, label_col

    def _encode_labels(self, y_group: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
        task = str(self.cfg.task).lower().strip()

        if task == "binary":
            y_bin = (y_group.astype(str) != "NORMAL").astype(int).to_numpy(dtype=np.int64)
            mapping = {"NORMAL": 0, "ATTACK": 1}
            return y_bin, mapping

        # multiclass: fixed order
        order = ["NORMAL", "DoS", "PROBE", "R2L", "U2R"]
        present = [c for c in order if c in set(y_group.astype(str).unique())]
        mapping = {lab: i for i, lab in enumerate(present)}
        y_mc = y_group.astype(str).map(mapping).to_numpy(dtype=np.int64)
        return y_mc, mapping

    # -----------------------------
    # Prepare
    # -----------------------------
    def prepare(self) -> None:
        df = self._load_raw_df()
        df, raw_label_col = self._clean_and_filter_df(df)

        y_group = df["_group_label"]
        y, mapping = self._encode_labels(y_group)
        self.label_to_id = mapping

        X_df = df.drop(columns=[raw_label_col, "_group_label"])

        # one-hot categorical if present (keeps structure consistent)
        cat_cols = [c for c in DEFAULT_CATEGORICAL_COLS if c in X_df.columns]
        if cat_cols:
            X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=False)

        X = X_df.to_numpy(dtype=np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y if len(np.unique(y)) > 1 else None
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

        # class weights (balanced)
        classes = np.unique(self.y_train)
        n_classes = int(len(np.unique(self.y_train)))
        if len(classes) >= 2:
            w = compute_class_weight(class_weight="balanced", classes=classes, y=self.y_train)
            w_full = np.zeros((max(int(classes.max()) + 1, n_classes),), dtype=np.float32)
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
        print(f"NSL-KDD prepared ✅ ({task})")
        print("scale:", self.cfg.scale)
        print("X_train:", None if self.X_train is None else self.X_train.shape)
        print("X_test :", None if self.X_test is None else self.X_test.shape)
        print("Label map:", self.label_to_id)
        if self.class_weights is not None:
            print("Class weights:", self.class_weights.tolist())
        if self.y_test is not None:
            u, c = np.unique(self.y_test, return_counts=True)
            print("y_test counts:", dict(zip(u.tolist(), c.tolist())))
