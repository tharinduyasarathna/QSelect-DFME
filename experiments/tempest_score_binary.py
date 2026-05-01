
# ============================================================
# FILE: experiments/tempest_score_binary.py
# ============================================================
# TEMPEST runner (tabular) for binary anomaly score teachers.
#
# Run:
#   python3 -m experiments.tempest_score_binary
# # ============================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from experiments.configs.tempest_score_binary_cfg import CFG
from experiments.utils.common_tabular import (
    sanitize_np,
    safe_nan_to_num_1d,
    safe_auc,
    make_loader,
)
from experiments.utils.builders_teachers_data import build_dm, build_teacher, csv_dir_for
from experiments.utils.io_csv_tempest import append_row
from experiments.utils.plot_rho_vs_queries_symlog import save_dataset_rho_vs_queries_plot

from experiments.utils.fidelity import score_fidelity_on_loader
from attacks.tempest import (
    TempestConfig,
    compute_public_stats,
    generate_queries_from_stats,
    train_student_tempest_from_cache,
)

torch.backends.cudnn.benchmark = True


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

    ms_cfg = CFG["milestones"]
    EDGE_QMIN = int(ms_cfg["edge_qmin"])
    EDGE_QMAX = int(ms_cfg["edge_qmax"])
    MILESTONES = sorted(set(
        list(ms_cfg["edge_budgets"]) +
        (list(ms_cfg["long_tail"]) if bool(ms_cfg["run_long_tail"]) else [])
    ))
    MAX_BUDGET = int(max(MILESTONES))

    tcfg = CFG["tempest"]
    GEN_MODE = str(tcfg["gen_mode"])
    ADV_NORM = str(tcfg["adv_norm"])
    EPOCHS = int(tcfg["epochs"])
    BS = int(tcfg["batch_size"])
    LR = float(tcfg["lr"])
    WD = float(tcfg["weight_decay"])
    VAR_EPS = float(tcfg["var_eps"])
    CLIP_STD = float(tcfg["clip_std"])
    TEACHER_CHUNK = int(tcfg["teacher_query_chunk"])

    EVAL_BATCH = int(CFG["eval"]["batch"])

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
        X_test  = sanitize_np(dm.X_test)
        y_train = np.asarray(dm.y_train)
        y_test  = np.asarray(dm.y_test)

        y_train_bin = (y_train != 0).astype(int)
        y_test_bin  = (y_test  != 0).astype(int)

        # teacher fit on normal-only
        X_train_norm = sanitize_np(X_train[y_train_bin == 0])
        d_in = int(X_train.shape[1])

        test_loader = make_loader(X_test, y_test, batch_size=EVAL_BATCH, shuffle=False)

        # public stats (data-free source)
        rng = np.random.RandomState(seed)
        n_pub = min(20_000, X_train.shape[0])
        pub_idx = rng.choice(np.arange(X_train.shape[0]), size=n_pub, replace=False)
        pub_stats = compute_public_stats(X_train[pub_idx])

        merged_curves = {}

        for vt in victims:
            print("\n------------------------------------")
            print("Victim:", vt)

            teacher = build_teacher(vt, d_in=d_in, device=DEVICE)
            teacher.fit(X_train_norm.astype(np.float32, copy=False))

            # teacher scores on test (flip if needed)
            t_scores = safe_nan_to_num_1d(teacher.score(X_test.astype(np.float32, copy=False)))
            auc_raw, _ = safe_auc(y_test_bin, t_scores)
            flipped = False
            if auc_raw == auc_raw and auc_raw < 0.5:
                flipped = True
                t_scores = -t_scores

            teacher_auroc, teacher_auprc = safe_auc(y_test_bin, t_scores)
            print(f"[Teacher] AUROC={teacher_auroc:.4f} AUPRC={teacher_auprc:.4f} (flip={int(flipped)})")

            def teacher_score_fn(x_t: torch.Tensor) -> torch.Tensor:
                x_np = x_t.detach().cpu().numpy().astype(np.float32, copy=False)
                x_np = sanitize_np(x_np)
                s_np = safe_nan_to_num_1d(teacher.score(x_np))
                if flipped:
                    s_np = -s_np
                return torch.from_numpy(s_np).to(x_t.device)

            # ---------------------------------------------------------
            # ONE MAX_BUDGET generation + ONE MAX_BUDGET teacher query
            # ---------------------------------------------------------
            print(f"\n[TEMPEST] generate Xq_full (MAX_BUDGET={MAX_BUDGET}) gen={GEN_MODE}")
            Xq_full = generate_queries_from_stats(
                n=int(MAX_BUDGET),
                stats=pub_stats,
                gen_mode=str(GEN_MODE),
                seed=int(seed),
                var_eps=float(VAR_EPS),
                clip_std=float(CLIP_STD),
            ).astype(np.float32, copy=False)

            print(f"[TEMPEST] query teacher once ({MAX_BUDGET} queries) ...")
            yT_full = np.zeros((int(MAX_BUDGET),), dtype=np.float32)
            t0 = time.time()
            with torch.no_grad():
                for i in range(0, int(MAX_BUDGET), TEACHER_CHUNK):
                    xb = torch.from_numpy(Xq_full[i:i + TEACHER_CHUNK]).to(device).float()
                    yb = teacher_score_fn(xb).reshape(-1).detach().cpu().numpy().astype(np.float32)
                    yT_full[i:i + yb.shape[0]] = yb
            t1 = time.time()
            sec_query = float(t1 - t0)
            print(f"[TEMPEST] teacher query done | sec={sec_query:.2f} | qps={MAX_BUDGET/max(sec_query,1e-9):.1f}")

            curve_q_used, curve_rho = [], []

            # ---------------------------------------------------------
            # milestone checkpoints (fresh student per budget)
            # ---------------------------------------------------------
            for budget in MILESTONES:
                student = nn.Sequential(
                    nn.Linear(d_in, 512), nn.ReLU(),
                    nn.Linear(512, 512), nn.ReLU(),
                    nn.Linear(512, 1)
                ).to(device)

                cfg = TempestConfig(
                    query_budget=int(budget),
                    gen_mode=str(GEN_MODE),
                    adv_norm=str(ADV_NORM),
                    epochs=int(EPOCHS),
                    batch_size=int(BS),
                    lr=float(LR),
                    weight_decay=float(WD),
                    seed=int(seed),
                    var_eps=float(VAR_EPS),
                    clip_std=float(CLIP_STD),
                )

                print(f"\n[CHECKPOINT] budget={budget} gen={GEN_MODE} norm={ADV_NORM}")
                stats = train_student_tempest_from_cache(
                    Xq_full=Xq_full,
                    yT_full=yT_full,
                    student=student,
                    device=device,
                    stats=pub_stats,
                    cfg=cfg,
                )

                fid = score_fidelity_on_loader(teacher_score_fn, student, test_loader, device)
                rho = float(fid["spearman_rho"])

                # AUROC of student scores
                with torch.no_grad():
                    X_test_t = torch.from_numpy(X_test).to(device)
                    bs2 = 65536
                    out = []
                    for i in range(0, X_test_t.size(0), bs2):
                        out.append(student(X_test_t[i:i + bs2]).detach().cpu().numpy())
                    s_scores = safe_nan_to_num_1d(np.concatenate(out, axis=0))
                student_auroc, student_auprc = safe_auc(y_test_bin, s_scores)

                curve_q_used.append(int(stats["query_budget"]))
                curve_rho.append(float(rho))

                run_tag = f"{ds}_{vt}_TEMPEST_TRAJ_max{MAX_BUDGET}_{GEN_MODE}_{ADV_NORM}"
                row = dict(
                    run_tag=run_tag,
                    run_datetime=RUN_DT,
                    dataset=ds,
                    victim_type=vt,
                    seed=int(seed),
                    d_in=int(d_in),
                    max_rows=int(max_rows),

                    budget_target=int(budget),
                    gen_mode=str(GEN_MODE),
                    adv_norm=str(ADV_NORM),
                    epochs=int(EPOCHS),
                    batch_size=int(BS),

                    query_budget=int(stats["query_budget"]),
                    extract_sec=float(stats["extract_sec"]),
                    queries_per_sec=float(stats["queries_per_sec"]),
                    ms_per_query=float(stats["ms_per_query"]),

                    teacher_auroc=float(teacher_auroc),
                    teacher_auprc=float(teacher_auprc),
                    student_auroc=float(student_auroc),
                    student_auprc=float(student_auprc),

                    fidelity_mse=float(fid["score_mse"]),
                    fidelity_mae=float(fid["score_mae"]),
                    fidelity_spearman=float(rho),
                )
                append_row(RESULTS_CSV, row)

                print(f"[SAVED] rho={rho:.4f} | AUROC_s={student_auroc:.4f} -> {RESULTS_CSV}")

            merged_curves[vt] = {"q_used": curve_q_used, "rho": curve_rho}

        save_dataset_rho_vs_queries_plot(
            out_dir=PLOT_DIR,
            dataset=ds,
            curves=merged_curves,
            tau_rho=float(TAU_RHO),
            edge_qmin=int(EDGE_QMIN),
            edge_qmax=int(EDGE_QMAX),
            linthresh=int(pcfg["linthresh"]),
            linscale=float(pcfg["linscale"]),
            xmax=int(MAX_BUDGET),
            fig_size=tuple(pcfg["fig_size"]),
            dpi=int(pcfg["dpi"]),
        )
        print(f"[MERGED PLOT SAVED] {PLOT_DIR}")

    print("\nDONE. CSV:", RESULTS_CSV)


if __name__ == "__main__":
    main()

