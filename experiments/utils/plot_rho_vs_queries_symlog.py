# ============================================================
# FILE: experiments/utils/plot_rho_vs_queries_symlog.py
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from .common_tabular import safe_slug


def _fmt_q(v: int) -> str:
    if v == 0:
        return "0"
    if v >= 1_000_000:
        return f"{int(v/1_000_000)}M"
    if v >= 1000:
        return f"{int(v/1000)}k"
    return str(v)


def save_dataset_rho_vs_queries_plot(
    out_dir: str,
    dataset: str,
    curves: dict,
    tau_rho: float,
    edge_qmin: int,
    edge_qmax: int,
    linthresh: int = 500,
    linscale: float = 1.2,
    xmax: int = 1_000_000,
    fig_size=(12.5, 5.8),
    dpi: int = 250,
):
    if not curves:
        return
    os.makedirs(out_dir, exist_ok=True)

    xticks = [
        0, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000,
        20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000
    ]

    fig = plt.figure(figsize=fig_size)
    ax = plt.gca()

    ax.axvspan(edge_qmin, edge_qmax, alpha=0.12)
    ax.axhline(float(tau_rho), linestyle="--", linewidth=1)

    y_all = []
    for victim, dat in curves.items():
        xs = list(map(int, dat.get("q_used", [])))
        ys = list(map(float, dat.get("rho", [])))
        if not xs:
            continue

        pairs = sorted(zip(xs, ys), key=lambda t: int(t[0]))
        xs = [int(p[0]) for p in pairs]
        ys = [float(p[1]) for p in pairs]

        if xs[0] != 0:
            xs = [0] + xs
            ys = [0.0] + ys

        y_all.append(np.asarray(ys, dtype=np.float32))
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=victim)

    if not y_all:
        plt.close(fig)
        return

    y_concat = np.concatenate(y_all, axis=0)
    y_min = float(np.nanmin(y_concat))
    y_max = float(np.nanmax(y_concat))
    if y_min >= 0.0:
        ax.set_ylim(bottom=0.0, top=min(1.0, max(0.05, y_max + 0.05)))
    else:
        ax.set_ylim(bottom=min(y_min - 0.05, -0.05), top=min(1.0, y_max + 0.05))

    ax.set_xscale("symlog", linthresh=linthresh, linscale=linscale)
    ax.set_xlim(left=0, right=max(int(xmax), int(edge_qmax)))
    ax.set_xticks(xticks)
    ax.set_xticklabels([_fmt_q(x) for x in xticks], rotation=90, ha="right")

    ax.set_ylabel("Behavioural fidelity (Spearman ρ)")
    ax.set_xlabel("Effective teacher queries used")
    ax.set_title(f"{dataset} — Fidelity vs Queries (merged victims)")
    ax.grid(True, which="both", linewidth=0.6, alpha=0.35)
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    base = f"{safe_slug(dataset)}__rho_vs_queries__merged_victims__symlog"
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, base + ".png"), dpi=dpi)
    plt.savefig(os.path.join(out_dir, base + ".pdf"))
    plt.close(fig)