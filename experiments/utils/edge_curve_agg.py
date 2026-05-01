# ============================================================
# FILE: experiments/utils/edge_curve_agg.py


from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

# Old format: target -> (used, rho)
EdgeRunTuple = Dict[int, Tuple[int, float]]

# New format: target -> {"used":..., "rho":..., ...}
EdgeRunDict = Dict[int, Dict[str, Any]]

# Accept either
EdgeRun = Union[EdgeRunTuple, EdgeRunDict]

# Output always uses (used, rho)
EdgeAvg = Dict[int, Tuple[int, float]]


def _extract_used_rho(v: Any) -> Optional[Tuple[int, float]]:
    """
    Normalize one milestone value into (used:int, rho:float).

    Supports:
      - tuple/list (used, rho)
      - dict with keys {"used","rho"} (extra keys allowed)

    Returns None if invalid or rho is NaN.
    """
    used = None
    rho = None

    # tuple/list (used, rho)
    if isinstance(v, (tuple, list)) and len(v) >= 2:
        used, rho = v[0], v[1]

    # dict {"used":..., "rho":...}
    elif isinstance(v, dict):
        used = v.get("used", None)
        rho = v.get("rho", None)

    else:
        return None

    if used is None or rho is None:
        return None

    try:
        used_i = int(used)
        rho_f = float(rho)
    except Exception:
        return None

    # drop NaN rhos
    if not (rho_f == rho_f):
        return None

    return used_i, rho_f


def avg_edge_runs(edge_runs: List[EdgeRun]) -> EdgeAvg:
    """
    Average per-milestone (target budget) across edge repeat runs.

    edge_runs: list of dicts (one per repeat), each:
      { target_budget -> (used_queries, rho) }
      OR
      { target_budget -> {"used": used_queries, "rho": rho, ...} }

    returns:
      { target_budget -> (avg_used_queries, avg_rho) }
    """
    if not edge_runs:
        return {}

    keys = sorted(set().union(*[set(d.keys()) for d in edge_runs]))
    out: EdgeAvg = {}

    for target in keys:
        pairs: List[Tuple[int, float]] = []
        for d in edge_runs:
            if target not in d:
                continue
            p = _extract_used_rho(d[target])
            if p is not None:
                pairs.append(p)

        if not pairs:
            continue

        useds = [float(u) for (u, _) in pairs]
        rhos = [float(r) for (_, r) in pairs]  # already NaN-filtered

        avg_used = int(round(float(np.mean(useds)))) if useds else int(target)
        avg_rho = float(np.mean(rhos)) if rhos else float("nan")

        if avg_rho == avg_rho:
            out[int(target)] = (avg_used, avg_rho)

    return out


def best_edge_runs(edge_runs: List[EdgeRun]) -> EdgeAvg:
    """
    Take BEST (max rho) per-milestone across EDGE repeats.

    edge_runs: list of dicts (one per repeat), each:
      { target_budget -> (used_queries, rho) }
      OR
      { target_budget -> {"used": used_queries, "rho": rho, ...} }

    returns:
      { target_budget -> (used_queries_of_best, best_rho) }
    """
    if not edge_runs:
        return {}

    keys = sorted(set().union(*[set(d.keys()) for d in edge_runs]))
    out: EdgeAvg = {}

    for target in keys:
        vals2: List[Tuple[int, float]] = []
        for d in edge_runs:
            if target not in d:
                continue
            p = _extract_used_rho(d[target])
            if p is not None:
                vals2.append(p)

        if not vals2:
            continue

        u_best, r_best = max(vals2, key=lambda t: t[1])
        out[int(target)] = (int(u_best), float(r_best))

    return out


def merge_edge_avg_into_traj(
    edge_avg: EdgeAvg,
    traj_q_used: List[int],
    traj_rho: List[float],
    edge_qmax: int,
    *,
    edge_x: str = "target",         # "target" | "avg_used"
    ensure_origin: bool = True,
    traj_skip_slack: int = 128,     # skip traj points slightly above edge_qmax
) -> Tuple[List[int], List[float]]:
    """
    Build a single curve for plotting:
      - for <= edge_qmax: use averaged EDGE points
      - for > edge_qmax (+ slack): use main trajectory points
    """
    xs: List[int] = []
    ys: List[float] = []

    # ---------- EDGE section ----------
    for target in sorted(edge_avg.keys()):
        if int(target) <= int(edge_qmax):
            avg_used, avg_rho = edge_avg[target]
            x = int(target) if edge_x == "target" else int(avg_used)
            xs.append(x)
            ys.append(float(avg_rho))

    # ---------- TRAJ section ----------
    cutoff = int(edge_qmax) + int(max(0, traj_skip_slack))

    for q, r in sorted(zip(traj_q_used, traj_rho), key=lambda t: int(t[0])):
        if int(q) <= cutoff:
            continue
        xs.append(int(q))
        ys.append(float(r))

    # origin point
    if ensure_origin and (not xs or xs[0] != 0):
        xs = [0] + xs
        ys = [0.0] + ys

    return xs, ys


def summarize_edge_avg(edge_avg: EdgeAvg) -> Dict[str, float]:
    """
    Small helper if you want to print a summary.
    """
    if not edge_avg:
        return {"count": 0.0, "mean_rho": float("nan")}
    rhos = [rho for (_, rho) in edge_avg.values() if rho == rho]
    return {"count": float(len(edge_avg)), "mean_rho": float(np.mean(rhos)) if rhos else float("nan")}