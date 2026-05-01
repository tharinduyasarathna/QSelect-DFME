"""
Auto-generate 6 LaTeX tables (one per dataset) from 3 CSVs.

"""

import os
import re
import numpy as np
import pandas as pd

# -------- paths (edit if needed) --------
CSV_PATHS = {
    "QSelDFME": "results/qselect_dfme_score_binary.csv",
    "TEMPEST": "results/tempest_score_binary.csv",
    "TabExtractor": "results/tabextractor_score_binary.csv",
}

OUT_DIR = "results/latex_tables"
BUDGET_2K = 2048

VICTIMS = ["pyod-AE", "pyod-VAE", "pyod-DeepSVDD", "DRocc", "NeuTraL-AD"]
METHOD_ORDER = ["QSelDFME", "TEMPEST", "TabExtractor"]
THRESHOLDS = (0.60, 0.75, 0.85)

# -------- best-per-column bolding directions --------
# True => higher is better; False => lower is better
BOLD_DIR = {
    "auc_at_2k": True,
    "rho_best_2k": True,     # bold only (NOT used for Winner)
    "qps_at_2k": False,      # LOWER QPS is better in your stealth/edge framing
    "q_ge_060": False,
    "q_ge_075": False,
    "q_ge_085": False,
}

# -------- Winner scoring (RQ3.2 weighted) --------
# IMPORTANT: rho_best_2k is intentionally NOT included here.
WIN_WEIGHTS = {
    "auc_at_2k": 1.0,     # higher better
    "qps_at_2k": 4.0,     # lower better (high weight)
    "q_ge_060": 1.0,      # lower better
    "q_ge_075": 2.0,      # lower better (main target)
    "q_ge_085": 1.0,      # lower better
}

# tie-break priority (after weighted ranks)
TIEBREAK = [
    "q_ge_075",
    "qps_at_2k",
    "auc_at_2k",
    "q_ge_085",
    "q_ge_060",
]


# ----------------------------
# Formatting (RAW values)
# ----------------------------
def _is_nan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))

def fmt_q(x):
    if _is_nan(x):
        return "-"
    return str(int(x))

def fmt_auc(x):
    if _is_nan(x):
        return "-"
    return f"{float(x):.3f}"

def fmt_rho(x):
    if _is_nan(x):
        return "-"
    return f"{float(x):.3f}"

def fmt_qps(x):
    if _is_nan(x):
        return "-"
    return f"{float(x):.0f}"

def bold_if(s, do_bold):
    if (not do_bold) or (s == "-"):
        return s
    return f"\\textbf{{{s}}}"


# ----------------------------
# Curve summarisation
# ----------------------------
def _row_at_budget(df, budget_ref):
    """
    Prefer LAST logged row with query_budget <= budget_ref.
    If none exist, fall back to NEAREST row.
    """
    if df is None or len(df) == 0:
        return None
    d = df.sort_values("query_budget")
    le = d[d["query_budget"] <= budget_ref]
    if len(le) > 0:
        return le.iloc[-1]
    i = int(np.argmin(np.abs(d["query_budget"].to_numpy() - budget_ref)))
    return d.iloc[i]

def summarize_curve(df, budget_ref=BUDGET_2K):
    if df is None or len(df) == 0:
        return dict(
            auc_at_2k=np.nan,
            rho_best_2k=np.nan,
            qps_at_2k=np.nan,
            q_ge_060=np.nan,
            q_ge_075=np.nan,
            q_ge_085=np.nan,
        )

    d = df.sort_values("query_budget")

    # best rho under <=2k
    d_2k = d[d["query_budget"] <= budget_ref]
    rho_best_2k = float(d_2k["fidelity_spearman"].max()) if len(d_2k) > 0 else np.nan

    # AUC/QPS at "2k point"
    r2k = _row_at_budget(d, budget_ref)
    if r2k is None:
        auc_at_2k = np.nan
        qps_at_2k = np.nan
    else:
        auc_at_2k = float(r2k.get("student_auroc", np.nan))
        qps_at_2k = float(r2k.get("queries_per_sec", np.nan))

    # thresholds: earliest q where rho >= t
    out = {}
    for t in THRESHOLDS:
        m = d["fidelity_spearman"] >= t
        out[t] = int(d.loc[m, "query_budget"].iloc[0]) if m.any() else np.nan

    return dict(
        auc_at_2k=auc_at_2k,
        rho_best_2k=rho_best_2k,
        qps_at_2k=qps_at_2k,
        q_ge_060=out[0.60],
        q_ge_075=out[0.75],
        q_ge_085=out[0.85],
    )


# ----------------------------
# Best-per-column (bolding)
# ----------------------------
def best_mask_for_metric(method_summaries, metric, higher_is_better):
    vals = {m: method_summaries[m].get(metric, np.nan) for m in method_summaries.keys()}
    valid = [(m, v) for m, v in vals.items() if not _is_nan(v)]
    if not valid:
        return {m: False for m in vals.keys()}

    vs = np.array([v for _, v in valid], dtype=np.float64)
    best_v = np.nanmax(vs) if higher_is_better else np.nanmin(vs)

    return {
        m: (not _is_nan(vals[m]) and np.isclose(vals[m], best_v, atol=1e-12))
        for m in vals.keys()
    }


# ----------------------------
# Overall Winner per victim (WEIGHTED for RQ3.2)
# ----------------------------
def _metric_ranks(method_summaries, metric, higher_is_better):
    """
    Dense ranking: 1 = best. NaNs get worst rank (999).
    """
    methods = list(method_summaries.keys())
    vals = {m: method_summaries[m].get(metric, np.nan) for m in methods}

    valid = [(m, float(v)) for m, v in vals.items() if not _is_nan(v)]
    if not valid:
        return {m: 999 for m in methods}

    valid_sorted = sorted(
        valid,
        key=(lambda t: (-t[1], t[0])) if higher_is_better else (lambda t: (t[1], t[0]))
    )

    ranks = {m: 999 for m in methods}
    rank = 1
    prev_v = None

    for m, v in valid_sorted:
        if prev_v is None:
            ranks[m] = rank
            prev_v = v
            continue
        if np.isclose(v, prev_v, atol=1e-12):
            ranks[m] = rank
        else:
            rank += 1
            ranks[m] = rank
            prev_v = v

    return ranks


def compute_overall_winner(method_summaries):
    """
    Weighted rank aggregation aligned to RQ3.2:
      - Only performance within <=2k counts.
      - If q_{rho>=tau} > 2048 → treated as failure (worst rank).
    """

    methods = list(method_summaries.keys())

    # Preprocess: enforce <=2k constraint on threshold metrics
    adjusted = {}

    for m in methods:
        s = method_summaries[m].copy()

        # If threshold achieved beyond 2k → invalidate
        for k in ["q_ge_060", "q_ge_075", "q_ge_085"]:
            qv = s.get(k, np.nan)
            if _is_nan(qv) or qv > BUDGET_2K:
                s[k] = np.nan   # treat as failure under RQ3.2

        adjusted[m] = s

    # compute ranks only for relevant metrics
    needed_metrics = set(WIN_WEIGHTS.keys()) | set(TIEBREAK)
    rank_maps = {}

    for metric in needed_metrics:

        if metric == "auc_at_2k":
            higher_is_better = True
        elif metric == "qps_at_2k":
            higher_is_better = False
        elif metric.startswith("q_ge_"):
            higher_is_better = False
        else:
            continue

        rank_maps[metric] = _metric_ranks(adjusted, metric, higher_is_better)

    # weighted rank score
    scores = {m: 0.0 for m in methods}

    for metric, weight in WIN_WEIGHTS.items():
        for m in methods:
            scores[m] += weight * rank_maps[metric][m]

    best_score = min(scores.values())
    candidates = [m for m in methods if np.isclose(scores[m], best_score, atol=1e-12)]

    if len(candidates) == 1:
        return candidates[0]

    # tie-break
    def tie_key(m):
        return tuple(rank_maps[metric][m] for metric in TIEBREAK) + (m,)

    return sorted(candidates, key=tie_key)[0]


# ----------------------------
# Build LaTeX table
# ----------------------------
def make_latex_table(dataset, victim_method_to_summary):

    label = "tab:" + re.sub(r"[^a-zA-Z0-9]+", "_", dataset).lower() + "_2k"

    header = f"""
\\begin{{table*}}[t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{6pt}}
\\renewcommand{{\\arraystretch}}{{1.18}}
\\caption{{{dataset}: low-budget extraction summary (budget cap $\\leq$2048 queries).
AUC and QPS are reported at the last logged point $\\leq$2048 (fallback: nearest).
Best $\\rho_{{\\le2k}}$ is the maximum Spearman value observed for any logged point with $q\\le2048$ (reported for reference; not used to select Winner).
For $q_{{\\rho\\ge\\tau}}$, smaller is better. For QPS, smaller is better (stealth).
Best values per victim across methods are bolded; \\textit{{Winner}} is computed via weighted rank aggregation (RQ3.2).}}
\\label{{{label}}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llrrrrrrl}}
\\toprule
Victim & Method & AUC@2k & Best $\\rho_{{\\le2k}}$ & QPS@2k & $q_{{\\rho\\ge0.60}}$ & $q_{{\\rho\\ge0.75}}$ & $q_{{\\rho\\ge0.85}}$ & Winner \\\\
\\midrule
""".strip("\n")

    body = []

    for v_i, victim in enumerate(VICTIMS):

        method_summaries = {m: victim_method_to_summary.get((victim, m), {}) for m in METHOD_ORDER}
        winner = compute_overall_winner(method_summaries)

        best_masks = {
            metric: best_mask_for_metric(method_summaries, metric, direction)
            for metric, direction in BOLD_DIR.items()
        }

        for m_i, method in enumerate(METHOD_ORDER):

            s = method_summaries[method]

            victim_cell = f"\\multirow{{3}}{{*}}{{{victim}}}" if m_i == 0 else ""
            winner_cell = f"\\multirow{{3}}{{*}}{{{winner}}}" if m_i == 0 else ""

            auc_str   = bold_if(fmt_auc(s.get("auc_at_2k")), best_masks["auc_at_2k"][method])
            rho2k_str = bold_if(fmt_rho(s.get("rho_best_2k")), best_masks["rho_best_2k"][method])
            qps_str   = bold_if(fmt_qps(s.get("qps_at_2k")), best_masks["qps_at_2k"][method])
            q60_str   = bold_if(fmt_q(s.get("q_ge_060")), best_masks["q_ge_060"][method])
            q75_str   = bold_if(fmt_q(s.get("q_ge_075")), best_masks["q_ge_075"][method])
            q85_str   = bold_if(fmt_q(s.get("q_ge_085")), best_masks["q_ge_085"][method])

            body.append(
                f"{victim_cell} & {method} & {auc_str} & {rho2k_str} & {qps_str} & "
                f"{q60_str} & {q75_str} & {q85_str} & {winner_cell} \\\\"
            )

        # line after every victim block
        if v_i != len(VICTIMS) - 1:
            body.append("\\midrule")

    footer = """
\\bottomrule
\\end{tabular}%
}
\\end{table*}
""".strip("\n")

    return header + "\n" + "\n".join(body) + "\n" + footer


# ----------------------------
# Main
# ----------------------------
def main():

    dfs = {}
    for method, path in CSV_PATHS.items():
        df = pd.read_csv(path)

        for col in ["query_budget", "fidelity_spearman", "student_auroc", "queries_per_sec"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        dfs[method] = df

    datasets = sorted(set().union(*[set(dfs[m]["dataset"].unique()) for m in dfs]))

    os.makedirs(OUT_DIR, exist_ok=True)

    for ds in datasets:
        victim_method_to_summary = {}

        for victim in VICTIMS:
            for method in METHOD_ORDER:
                sub = dfs[method][
                    (dfs[method]["dataset"] == ds) &
                    (dfs[method]["victim_type"] == victim)
                ].copy()

                victim_method_to_summary[(victim, method)] = summarize_curve(sub, budget_ref=BUDGET_2K)

        tex = make_latex_table(ds, victim_method_to_summary)

        filename = os.path.join(
            OUT_DIR,
            f"table_{re.sub(r'[^a-zA-Z0-9]+','_', ds)}_2k.tex"
        )

        with open(filename, "w", encoding="utf-8") as f:
            f.write(tex)

        print("Wrote:", filename)

    print("\nDone.")
    print("Required in preamble:")
    print("  \\\\usepackage{booktabs}")
    print("  \\\\usepackage{multirow}")


if __name__ == "__main__":
    main()