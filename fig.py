import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CSV_PATHS = {
    "QSelect": "results/qselect_dfme_score_binary.csv",
    "TEMPEST": "results/tempest_score_binary.csv",
    "TabExtractor": "results/tabextractor_score_binary.csv",
}

OUT_DIR = "final_figures"
os.makedirs(OUT_DIR, exist_ok=True)

BUDGET = 2048
RHO_TARGET = 0.75

METHOD_ORDER = ["QSelect", "TEMPEST", "TabExtractor"]
MODEL_ORDER = ["pyod-AE", "pyod-VAE", "pyod-DeepSVDD", "DRocc", "NeuTraL-AD"]
DATASET_ORDER = [
    "CIC-IDS2017",
    "CICIoT2023",
    "InSDN_DatasetCSV",
    "SDN-IoT",
    "NSL-KDD",
    "UNSW-NB15",
]

# =========================
# LOAD DATA
# =========================
dfs = {}
for name, path in CSV_PATHS.items():
    df = pd.read_csv(path)

    for col in ["query_budget", "fidelity_spearman", "student_auroc", "queries_per_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["victim_type"] = df["victim_type"].astype(str).str.strip()

    dfs[name] = df

# Use only datasets that actually exist in the CSVs, preserving preferred order
all_datasets = set()
for df in dfs.values():
    all_datasets.update(df["dataset"].dropna().unique().tolist())

datasets = [d for d in DATASET_ORDER if d in all_datasets]
remaining = sorted(all_datasets - set(datasets))
datasets.extend(remaining)

# =========================
# HELPERS
# =========================
def get_subset(df, dataset, victim):
    sub = df[(df["dataset"] == dataset) & (df["victim_type"] == victim)].copy()
    return sub.sort_values("query_budget")


def get_rho_at_budget(df, budget=BUDGET):
    """
    Return fidelity_spearman at the last logged point <= budget.
    Fallback: nearest logged point if none <= budget.
    """
    if df.empty:
        return np.nan

    d = df[df["query_budget"] <= budget]
    if len(d) > 0:
        return float(d.iloc[-1]["fidelity_spearman"])

    idx = (df["query_budget"] - budget).abs().argsort().iloc[0]
    return float(df.loc[idx, "fidelity_spearman"])


def get_q_at_threshold(df, tau):
    """
    Return the first query budget where fidelity_spearman >= tau.
    If never reached, return np.nan.
    """
    if df.empty:
        return np.nan

    m = df["fidelity_spearman"] >= tau
    if m.any():
        return float(df.loc[m, "query_budget"].iloc[0])

    return np.nan


# =========================
# BUILD MATRICES
# =========================
delta_matrix = np.full((len(datasets), len(MODEL_ORDER)), np.nan)
q_matrix = np.full((len(datasets), len(MODEL_ORDER)), np.nan)

for i, ds in enumerate(datasets):
    for j, model in enumerate(MODEL_ORDER):
        # QSelect rho@2k
        qselect_sub = get_subset(dfs["QSelect"], ds, model)
        rho_qselect = get_rho_at_budget(qselect_sub, BUDGET)

        # Baseline rho@2k
        tempest_sub = get_subset(dfs["TEMPEST"], ds, model)
        tab_sub = get_subset(dfs["TabExtractor"], ds, model)

        rho_tempest = get_rho_at_budget(tempest_sub, BUDGET)
        rho_tab = get_rho_at_budget(tab_sub, BUDGET)

        best_baseline = np.nanmax([rho_tempest, rho_tab])
        if np.isfinite(rho_qselect) and np.isfinite(best_baseline):
            delta_matrix[i, j] = rho_qselect - best_baseline

        # QSelect query efficiency at rho >= 0.75
        q_val = get_q_at_threshold(qselect_sub, RHO_TARGET)
        q_matrix[i, j] = q_val if np.isfinite(q_val) else np.nan

# =========================
# FIGURE 1: Δρ@2k HEATMAP
# =========================
fig, ax = plt.subplots(figsize=(10, 5.5))
im = ax.imshow(delta_matrix, aspect="auto")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"$\Delta \rho@2k$")

ax.set_xticks(range(len(MODEL_ORDER)))
ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right")
ax.set_yticks(range(len(datasets)))
ax.set_yticklabels(datasets)

ax.set_xlabel("Teacher model")
ax.set_ylabel("Dataset")
ax.set_title(r"$\Delta \rho@2k$ (QSelect - Best Baseline)")

for i in range(len(datasets)):
    for j in range(len(MODEL_ORDER)):
        val = delta_matrix[i, j]
        text = "-" if np.isnan(val) else f"{val:.2f}"
        ax.text(j, i, text, ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_delta_rho_2k.pdf"), dpi=300, bbox_inches="tight")
plt.close()

# =========================
# FIGURE 2: q_{rho>=0.75} HEATMAP
# =========================
fig, ax = plt.subplots(figsize=(10, 5.5))
im = ax.imshow(q_matrix, aspect="auto")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"Queries to reach $\rho \geq 0.75$")

ax.set_xticks(range(len(MODEL_ORDER)))
ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right")
ax.set_yticks(range(len(datasets)))
ax.set_yticklabels(datasets)

ax.set_xlabel("Teacher model")
ax.set_ylabel("Dataset")
ax.set_title(r"Query efficiency: $q_{\rho \geq 0.75}$ (QSelect)")

for i in range(len(datasets)):
    for j in range(len(MODEL_ORDER)):
        val = q_matrix[i, j]
        text = "-" if np.isnan(val) else str(int(val))
        ax.text(j, i, text, ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_q75_heatmap.pdf"), dpi=300, bbox_inches="tight")
plt.close()

print("Done. Figures saved in:", OUT_DIR)
print(" - fig_delta_rho_2k.pdf")
print(" - fig_q75_heatmap.pdf")