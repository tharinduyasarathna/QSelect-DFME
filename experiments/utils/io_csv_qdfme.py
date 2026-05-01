# ============================================================
# FILE: experiments/utils/io_csv_qdfme.py
# ============================================================

import os
import csv

CSV_COLUMNS = [
    "run_tag","run_datetime","dataset","victim_type","seed","d_in","max_rows",
    "budget_target","budget_cap_actual","steps","pool_size","teacher_batch_size",
    "use_query_selection","prefilter_ratio","pca_dim",
    "student_steps_per_round","replay_ratio","replay_quantiles",
    "rank_loss_weight","rank_pairs",
    "refine_x_steps",
    "query_budget_expected","query_budget",
    "extract_sec","queries_per_sec","ms_per_query",
    "teacher_auroc","teacher_auprc",
    "student_auroc","student_auprc",
    "fidelity_mse","fidelity_mae","fidelity_spearman",
    "tau_rho","eps_auroc",
    "meets_rho","meets_both",
    "qmin_rho","qmin_both",
    "edge_qmin","edge_qmax",
    "within_edge_window",
    "qmin_rho_within_edge",
    "qmin_both_within_edge",
]


def append_row(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})