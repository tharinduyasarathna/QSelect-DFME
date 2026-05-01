# ============================================================
# FILE: experiments/utils/builders_teachers_data.py
# ============================================================

import os
import inspect
import numpy as np

from data_loader.cicids2017_loader import CICIDSConfig, CICIDS2017DataModule
from data_loader.insdn_loader import InSDNConfig, InSDNDataModule
from data_loader.aseados_sdn_iot_loader import ASEADOSConfig, ASEADOSDataModule
from data_loader.unswnb15_loader import UNSWNB15Config, UNSWNB15DataModule
from data_loader.ciciot23_loader import CICIoT23Config, CICIoT23DataModule
from data_loader.nslkdd_loader import NSLKDDConfig, NSLKDDDataModule

from models.drocc import DROCCDetector, DROCCConfig
from models.neutralad import NeuTraLADDetector, NeuTraLADConfig

from .common_tabular import safe_nan_to_num_1d


def csv_dir_for(ds_name: str) -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../data"))
    if ds_name in ("SDN-IoT", "SDN_IoT"):
        return os.path.join(base, "SDN-IoT")
    if ds_name == "CIC-IDS2017":
        return os.path.join(base, "CIC-IDS2017")
    if ds_name == "InSDN_DatasetCSV":
        return os.path.join(base, "InSDN_DatasetCSV")
    if ds_name == "UNSW-NB15":
        return os.path.join(base, "UNSW-NB15")
    if ds_name == "CICIoT2023":
        return os.path.join(base, "CICIoT2023")
    if ds_name == "NSL-KDD":
        return os.path.join(base, "NSL-KDD")
    raise ValueError(ds_name)


def build_dm(dataset_name: str, seed: int, batch_size: int, csv_dir: str, max_rows=None):
    name = dataset_name.strip()

    if name == "CIC-IDS2017":
        cfg = CICIDSConfig(
            csv_dir=csv_dir, task="binary", label_col="Label",
            test_size=0.25, random_state=seed, batch_size=batch_size,
            max_rows=max_rows, scale="zscore"
        )
        return CICIDS2017DataModule(cfg)

    if name == "InSDN_DatasetCSV":
        cfg = InSDNConfig(
            csv_dir=csv_dir, task="binary", label_col="Label",
            test_size=0.25, random_state=seed, batch_size=batch_size,
            max_rows=max_rows, scale="zscore"
        )
        return InSDNDataModule(cfg)

    if name in ("SDN-IoT", "SDN_IoT"):
        cfg = ASEADOSConfig(
            csv_dir=csv_dir, task="binary", label_col="Label",
            test_size=0.25, random_state=seed, batch_size=batch_size,
            max_rows=max_rows, scale="zscore",
            onehot_protocol=False,
        )
        return ASEADOSDataModule(cfg)

    if name == "UNSW-NB15":
        cfg = UNSWNB15Config(
            csv_dir=csv_dir, task="binary", label_col="label",
            test_size=0.25, random_state=seed, batch_size=batch_size,
            max_rows=max_rows, scale="zscore"
        )
        return UNSWNB15DataModule(cfg)

    if name == "CICIoT2023":
        cfg = CICIoT23Config(
            csv_dir=csv_dir, task="binary", label_col="label",
            test_size=0.25, random_state=seed, batch_size=batch_size,
            max_rows=max_rows, scale="zscore"
        )
        return CICIoT23DataModule(cfg)

    if name == "NSL-KDD":
        cfg = NSLKDDConfig(
            csv_dir=csv_dir, task="binary", label_col="label",
            test_size=0.25, random_state=seed, batch_size=batch_size,
            max_rows=max_rows, scale="zscore"
        )
        return NSLKDDDataModule(cfg)

    raise ValueError(f"Unknown dataset {dataset_name}")


def build_teacher(victim_type: str, d_in: int, device: str):
    vt = victim_type.strip()

    if vt == "DRocc":
        return DROCCDetector(
            d_in,
            DROCCConfig(
                hid_dim=256, epochs=20, batch_size=1024, lr=1e-3,
                lamda=1.0, radius=0.2, gamma=2.0,
                ascent_step_size=1e-3, ascent_num_steps=50
            ),
            device=device,
        )

    if vt == "NeuTraL-AD":
        return NeuTraLADDetector(
            d_in,
            NeuTraLADConfig(
                epochs=50, batch_size=1024, lr=1e-3,
                n_transforms=4, n_layers=3, trans_type="mlp", temperature=0.07
            ),
            device=device,
        )

    if vt in ("pyod-AE", "pyod-VAE", "pyod-DeepSVDD"):
        try:
            from pyod.models.auto_encoder import AutoEncoder as PyODAE
            from pyod.models.vae import VAE as PyODVAE
            from pyod.models.deep_svdd import DeepSVDD as PyODDeepSVDD
        except Exception as e:
            raise ImportError("PyOD required. Install: python3 -m pip install -U pyod") from e

        def instantiate(cls, kwargs):
            sig = inspect.signature(cls.__init__)
            params = sig.parameters
            filtered = {k: v for k, v in kwargs.items() if k in params}
            return cls(**filtered)

        class PyODWrap:
            def __init__(self, model):
                self.model = model

            def fit(self, X):
                self.model.fit(X)
                return {}

            def score(self, X):
                if hasattr(self.model, "decision_function"):
                    return safe_nan_to_num_1d(self.model.decision_function(X))
                return np.zeros((X.shape[0],), dtype=np.float32)

        if vt == "pyod-AE":
            return PyODWrap(instantiate(PyODAE, dict(epoch_num=10, batch_size=256, contamination=0.01, verbose=0)))
        if vt == "pyod-VAE":
            return PyODWrap(instantiate(PyODVAE, dict(epoch_num=15, batch_size=128, contamination=0.01, verbose=0)))
        if vt == "pyod-DeepSVDD":
            return PyODWrap(instantiate(PyODDeepSVDD, dict(n_features=d_in, epoch_num=10, batch_size=256, contamination=0.01, verbose=0)))

    raise ValueError(f"Unknown victim_type={victim_type}")