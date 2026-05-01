import numpy as np
from typing import List, Tuple

def agg_mean_std(values: List[float]) -> Tuple[float, float, int]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    if v.size == 1:
        return float(v.mean()), 0.0, 1
    return float(v.mean()), float(v.std(ddof=1)), int(v.size)