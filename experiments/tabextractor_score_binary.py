# ============================================================
# FILE: experiments/tabextractor_score_binary.py
# ============================================================
# TabExtractor runner (numerical-only) for binary anomaly-score teachers.
#
# Run:
#   python3 -m experiments.tabextractor_score_binary
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from experiments.configs.tabextractor_score_binary_cfg import CFG
from experiments.utils.common_tabular import (
    sanitize_np,
    stratified_subsample,
    safe_nan_to_num_1d,
    safe_auc,
    make_loader,
)
from experiments.utils.builders_teachers_data import build_dm, build_teacher, csv_dir_for
from experiments.utils.io_csv_tabextractor import append_row
from experiments.utils.plot_rho_vs_queries_symlog import save_dataset_rho_vs_queries_plot
from experiments.utils.fidelity import score_fidelity_on_loader

from attacks.tabextractor import (
    TabExtractorConfig,
    TabularGeneratorNum,
    CTTClone,
    train_student_tabextractor,  # <-- THIS EXISTS
)

torch.backends.cudnn.benchmark = True


class CloneScoreWrapper(nn.Module):
    """Wrap clone logits -> score (use class-1 logit as the extracted score)."""
    def __init__(self, clone: nn.Module):
        super().__init__()
        self.clone = clone

    def forward(self, x):
        return self.clone(x)[:, 1].reshape(-1)


def main():
    seed = int(CFG["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(DEVICE)
    print("Device:", device)

    datasets = list(CFG["datasets"])
    victims = list(CFG["victims"])
    max_rows = int(CFG["max_rows"])

    # milestones
    ms = list(CFG["milestones"]["budgets"])
    EDGE_QMIN = int(CFG["milestones"]["edge_qmin"])
    EDGE_QMAX = int(CFG["milestones"]["edge_qmax"])
    MILESTONES = sorted(set(int(x) for x in ms))
    MAX_QUERIES = int(CFG["tabextractor"]["max_queries"])

    eval_cfg = CFG["eval"]
    EVAL_BATCH = int(eval_cfg["batch"])
    EVAL_CHUNK = int(eval_cfg["eval_chunk"])
    MAX_EVAL_TEST = int(eval_cfg["max_eval_test"])
    MAX_FID_TEST = int(eval_cfg["max_fid_test"])

    tx = CFG["tabextractor"]
    TRAIN_BS = int(tx["train_bs"])
    THRESH_MODE = str(tx["label_threshold_mode"])
    USE_BOUNDS = bool(tx["use_feat_bounds"])
    USE_COMPILE = bool(tx["use_compile"])

    # iterations needed to reach max queries in ONE run
    MAX_ITERS = int(np.ceil(float(MAX_QUERIES) / float(TRAIN_BS)))
    LOG_EVERY = max(10, int(MAX_ITERS * float(tx["log_every_frac"])))

    pcfg = CFG["plot"]
    PLOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", pcfg["out_dir"]))
    TAU_RHO = float(pcfg["tau_rho"])

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RESULTS_CSV = os.path.join(PROJECT_ROOT, CFG["results"]["csv_path"])
    RUN_DT = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for ds in datasets:
        print("\n====================================")
        print("Dataset:", ds)
        CSV_DIR = csv_dir_for(ds)
        print("CSV_DIR:", CSV_DIR)

        dm = build_dm(ds, seed=seed, batch_size=EVAL_BATCH, csv_dir=CSV_DIR, max_rows=max_rows)
        dm.prepare()
        dm.summary()
        _ = dm.loaders()

        X_train = sanitize_np(dm.X_train)
        X_test = sanitize_np(dm.X_test)
        y_train = np.asarray(dm.y_train)
        y_test = np.asarray(dm.y_test)

        # eval subsamples (speed only)
        X_test_eval, y_test_eval = stratified_subsample(X_test, y_test, MAX_EVAL_TEST, seed=seed + 123)
        y_test_eval_bin = (y_test_eval != 0).astype(int)

        X_test_fid, y_test_fid = stratified_subsample(X_test, y_test, MAX_FID_TEST, seed=seed + 456)
        fid_loader = make_loader(X_test_fid, y_test_fid, batch_size=EVAL_BATCH, shuffle=False)

        y_train_bin = (y_train != 0).astype(int)
        X_train_norm = sanitize_np(X_train[y_train_bin == 0])
        d_in = int(X_train.shape[1])

        feat_lo = torch.from_numpy(np.min(X_train, axis=0).astype(np.float32)).to(device)
        feat_hi = torch.from_numpy(np.max(X_train, axis=0).astype(np.float32)).to(device)

        merged_curves = {}

        for vt in victims:
            print("\n------------------------------------")
            print("Victim:", vt)

            teacher = build_teacher(vt, d_in=d_in, device=DEVICE)
            teacher.fit(X_train_norm.astype(np.float32, copy=False))

            # teacher eval (flip if inverted)
            t_scores_eval = safe_nan_to_num_1d(teacher.score(X_test_eval.astype(np.float32, copy=False)))
            auc_raw, _ = safe_auc(y_test_eval_bin, t_scores_eval)

            flipped = False
            if auc_raw == auc_raw and auc_raw < 0.5:
                flipped = True
                t_scores_eval = -t_scores_eval

            teacher_auroc, teacher_auprc = safe_auc(y_test_eval_bin, t_scores_eval)
            print(f"[Teacher] AUROC={teacher_auroc:.4f} AUPRC={teacher_auprc:.4f} (flip={int(flipped)})")

            def teacher_score_fn(x_t: torch.Tensor) -> torch.Tensor:
                x_np = x_t.detach().cpu().numpy().astype(np.float32, copy=False)
                x_np = sanitize_np(x_np)
                s_np = safe_nan_to_num_1d(teacher.score(x_np))
                if flipped:
                    s_np = -s_np
                return torch.from_numpy(s_np).to(x_t.device)

            clone = CTTClone(d_in=d_in, n_classes=2, d_embed=128, n_heads=4, n_layers=2, dropout=0.1).to(device)
            gen = TabularGeneratorNum(d_in=d_in, z_dim=64, hidden=(256, 256)).to(device)

            if USE_COMPILE and hasattr(torch, "compile"):
                try:
                    clone = torch.compile(clone)
                    gen = torch.compile(gen)
                except Exception:
                    pass

            curve_q_used, curve_rho = [], []

            # because callback doesn’t tell “target”, we track it here
            milestones_left = list(MILESTONES)

            def on_milestone(event: dict):
                nonlocal milestones_left

                used = int(event["used"])
                stats = dict(event.get("stats", {}))

                # the target that was crossed
                target = int(milestones_left[0]) if milestones_left else used
                if milestones_left:
                    milestones_left.pop(0)

                # Student AUROC (use clone class-1 logit)
                with torch.no_grad():
                    clone.eval()
                    X_eval_t = torch.from_numpy(X_test_eval).to(device)
                    out = []
                    for i in range(0, X_eval_t.size(0), EVAL_CHUNK):
                        logits = clone(X_eval_t[i:i + EVAL_CHUNK])
                        out.append(logits[:, 1].detach().float().cpu().numpy())
                    s_scores = safe_nan_to_num_1d(np.concatenate(out, axis=0))

                student_auroc, student_auprc = safe_auc(y_test_eval_bin, s_scores)

                # Fidelity (Spearman rho) on fid set
                fid = score_fidelity_on_loader(
                    teacher_score_fn,
                    CloneScoreWrapper(clone).to(device).eval(),
                    fid_loader,
                    device,
                )
                rho = float(fid["spearman_rho"])

                curve_q_used.append(int(used))
                curve_rho.append(float(rho))

                run_tag = f"{ds}_{vt}_TabExtractor_TRAJ_max{MAX_QUERIES}_bs{TRAIN_BS}"
                row = dict(
                    run_tag=run_tag,
                    run_datetime=RUN_DT,
                    dataset=ds,
                    victim_type=vt,
                    seed=int(seed),
                    d_in=int(d_in),
                    max_rows=int(max_rows),

                    budget_target=int(target),
                    iterations=int(MAX_ITERS),
                    batch_size=int(TRAIN_BS),

                    query_budget=int(used),
                    extract_sec=float(stats.get("extract_sec", 0.0)),
                    queries_per_sec=float(stats.get("queries_per_sec", 0.0)),
                    ms_per_query=float(stats.get("ms_per_query", 0.0)),

                    teacher_auroc=float(teacher_auroc),
                    teacher_auprc=float(teacher_auprc),
                    student_auroc=float(student_auroc),
                    student_auprc=float(student_auprc),

                    fidelity_mse=float(fid["score_mse"]),
                    fidelity_mae=float(fid["score_mae"]),
                    fidelity_spearman=float(rho),
                )
                append_row(RESULTS_CSV, row)

                print(f"[MILESTONE] target={target} used={used} rho={rho:.4f} AUROC_s={student_auroc:.4f}")

            cfg = TabExtractorConfig(
                iterations=int(MAX_ITERS),
                batch_size=int(TRAIN_BS),
                seed=int(seed),
                label_threshold_mode=str(THRESH_MODE),
                use_feat_bounds=bool(USE_BOUNDS),
                log_every=int(LOG_EVERY),

                lr_clone=1e-3,
                lr_gen=1e-3,
                clone_steps=1,
                gen_steps=1,           # <-- enable generator update
                balance_weight=0.2,    # tune (0.1~0.5)
                entropy_weight=0.1,    # tune (0.0~0.2)
                grad_clip=1.0,

                milestones=list(MILESTONES),
                milestone_callback=on_milestone,
            )

            print(
                f"\n[TRAJ RUN] max_queries={MAX_QUERIES} -> iters={MAX_ITERS} bs={TRAIN_BS} "
                f"| milestones={len(MILESTONES)}"
            )

            _ = train_student_tabextractor(
                teacher_score_fn=teacher_score_fn,
                clone=clone,
                generator=gen,
                device=device,
                cfg=cfg,
                feat_lo=feat_lo,
                feat_hi=feat_hi,
            )

            merged_curves[vt] = {"q_used": curve_q_used, "rho": curve_rho}

            del clone, gen
            if device.type == "cuda":
                torch.cuda.empty_cache()

        save_dataset_rho_vs_queries_plot(
            out_dir=PLOT_DIR,
            dataset=ds,
            curves=merged_curves,
            tau_rho=float(TAU_RHO),
            edge_qmin=int(EDGE_QMIN),
            edge_qmax=int(EDGE_QMAX),
            linthresh=int(pcfg["linthresh"]),
            linscale=float(pcfg["linscale"]),
            xmax=int(MAX_QUERIES),
            fig_size=tuple(pcfg["fig_size"]),
            dpi=int(pcfg["dpi"]),
        )
        print(f"[MERGED PLOT SAVED] {PLOT_DIR}")

    print("\nDONE. CSV:", RESULTS_CSV)


if __name__ == "__main__":
    main()

