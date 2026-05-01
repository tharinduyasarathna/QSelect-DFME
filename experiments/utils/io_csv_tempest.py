# ============================================================
# FILE: experiments/utils/io_csv_tempest.py
# ============================================================

import os
import csv

CSV_COLUMNS = [
    "run_tag","run_datetime","dataset","victim_type","seed","d_in","max_rows",
    "budget_target","gen_mode","adv_norm","epochs","batch_size",
    "query_budget","extract_sec","queries_per_sec","ms_per_query",
    "teacher_auroc","teacher_auprc","student_auroc","student_auprc",
    "fidelity_mse","fidelity_mae","fidelity_spearman",
]

def append_row(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})