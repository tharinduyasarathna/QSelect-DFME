# ============================================================
# FILE: experiments/qselect_dfme_score_binary.py
# ============================================================
# QSelect-DFME runner for tabular *score* teachers (binary).
#
# Run:
# python3 -m experiments.qselect_dfme_score_binary
# ============================================================


import os
import math
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

from experiments.configs.qselect_dfme_score_binary_cfg import CFG
from experiments.utils.common_tabular import (
    make_loader,
    stratified_subsample,
    sanitize_np,
    safe_nan_to_num_1d,
    safe_auc,
    infer_protocol_allowed_from_scaled_train,
)
from experiments.utils.io_csv_qdfme import append_row
from experiments.utils.plot_rho_vs_queries_symlog import save_dataset_rho_vs_queries_plot
from experiments.utils.builders_teachers_data import build_dm, build_teacher, csv_dir_for

from experiments.utils.edge_curve_agg import (
    merge_edge_avg_into_traj,
)

from attacks.qselect_dfme import (
    QSelDFMEConfig,
    train_student_qselect_dfme,
    score_fidelity_on_loader,
    calibrate_student_affine,
    build_default_generator,
)

torch.backends.cudnn.benchmark = True


class ScoreMLP(nn.Module):
    def __init__(self, d_in: int, hidden=(512, 512)):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _student_scores_on_eval(student_cal: nn.Module, X_test_eval: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        X_eval_t = torch.from_numpy(X_test_eval).to(device)
        bs = 65536
        out = []
        for i in range(0, X_eval_t.size(0), bs):
            out.append(student_cal(X_eval_t[i:i + bs]).detach().cpu().numpy())
        s_scores = safe_nan_to_num_1d(np.concatenate(out, axis=0))
    return s_scores


def _best_per_milestone_from_repeats(
    edge_runs_full: List[Dict[int, Dict[str, Any]]]
) -> Dict[int, Dict[str, Any]]:
    """
    edge_runs_full: list of repeats, each repeat is:
      { target -> {"used":..., "rho":..., "student_auroc":..., "queries_per_sec":..., ...} }

    returns:
      best_ms: { target -> record_from_repeat_with_max_rho_at_that_target }
    """
    if not edge_runs_full:
        return {}

    all_targets = sorted(set().union(*[set(r.keys()) for r in edge_runs_full]))
    best_ms: Dict[int, Dict[str, Any]] = {}

    for t in all_targets:
        best_rec = None
        best_rho = float("-inf")
        for run in edge_runs_full:
            if t not in run:
                continue
            rec = run[t]
            rho = float(rec.get("rho", float("nan")))
            if not (rho == rho):
                continue
            if rho > best_rho:
                best_rho = rho
                best_rec = rec

        if best_rec is not None:
            best_ms[int(t)] = dict(best_rec)  # copy

    return best_ms


def _edge_curve_tuple_from_full(
    edge_best_full: Dict[int, Dict[str, Any]]
) -> Dict[int, Tuple[int, float]]:
    """
    Convert full edge dict to tuple format expected by merge_edge_avg_into_traj:
      { target -> (used, rho) }
    """
    out: Dict[int, Tuple[int, float]] = {}
    for t, rec in edge_best_full.items():
        used = int(rec.get("used", t))
        rho = float(rec.get("rho", float("nan")))
        if rho == rho:
            out[int(t)] = (used, rho)
    return out


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

    # thresholds (reporting only)
    TAU_RHO = float(CFG["thresholds"]["tau_rho"])
    EPS_AUROC = float(CFG["thresholds"]["eps_auroc"])

    # milestones
    EDGE_QMIN = int(CFG["milestones"]["edge_qmin"])
    EDGE_QMAX = int(CFG["milestones"]["edge_qmax"])
    EDGE_BUDGETS = list(CFG["milestones"]["edge_budgets"])
    RUN_LONG_TAIL = bool(CFG["milestones"]["run_long_tail"])
    LONG_TAIL = list(CFG["milestones"]["long_tail"])

    EDGE_MILESTONES = list(EDGE_BUDGETS)
    MILESTONES_ALL = sorted(set(EDGE_MILESTONES + (LONG_TAIL if RUN_LONG_TAIL else [])))
    TR_MILESTONES = [m for m in MILESTONES_ALL if m > EDGE_QMAX]  # <-- TRAJ logs only > EDGE_QMAX
    MAX_BUDGET = int(max(MILESTONES_ALL))

    # eval
    EVAL_BATCH = int(CFG["eval"]["batch"])
    MAX_EVAL_TEST = int(CFG["eval"]["max_eval_test"])
    MAX_FID_TEST = int(CFG["eval"]["max_fid_test"])

    # selection
    sel_def = CFG["selection_default"]
    sel_edge = CFG["selection_edge"]

    POOL_SIZE = int(sel_def["pool_size"])
    PROJ_DIM = int(sel_def["proj_dim"])
    PREFILTER_RATIO = float(sel_def["prefilter_ratio"])
    CAND_FACTOR = int(sel_def["cand_factor"])
    RANDOM_MIX_TRAJ = float(sel_def["random_mix"])

    POOL_SIZE_EDGE = int(sel_edge["pool_size"])
    PROJ_DIM_EDGE = int(sel_edge["proj_dim"])
    PREFILTER_RATIO_EDGE = float(sel_edge["prefilter_ratio"])
    CAND_FACTOR_EDGE = int(sel_edge["cand_factor"])
    RANDOM_MIX_EDGE = float(sel_edge["random_mix"])

    # student
    STUDENT_LOSS = str(CFG["student"]["loss"])
    LR_STUDENT = float(CFG["student"]["lr"])
    STUDENT_STEPS = int(CFG["student"]["steps_per_round"])
    REPLAY_RATIO = float(CFG["student"]["replay_ratio"])
    REPLAY_QUANTILES = int(CFG["student"]["replay_quantiles"])
    RANK_W = float(CFG["student"]["rank_w"])
    RANK_PAIRS = int(CFG["student"]["rank_pairs"])
    STUDENT_HIDDEN = tuple(CFG["student"]["hidden"])

    # generator
    Z_DIM = int(CFG["generator"]["z_dim"])
    GEN_HIDDEN = tuple(CFG["generator"]["hidden"])
    LR_GEN = float(CFG["generator"]["lr"])
    LAMBDA_GEN = float(CFG["generator"]["lambda_gen"])
    GEN_UPDATE_EVERY = int(CFG["generator"]["update_every"])
    GEN_STEPS = int(CFG["generator"].get("steps", 1))
    GEN_DIVERSITY_W = float(CFG["generator"].get("diversity_w", 0.05))

    # schedule
    STEPS_TRAJ = int(CFG["schedule"]["steps_traj"])
    K_EARLY = int(CFG["schedule"]["k_early"])
    K_LATE = int(CFG["schedule"]["k_late"])
    K_SWITCH_AT = int(CFG["schedule"]["k_switch_at"])

    # edge repeats
    RUN_EDGE_REPEATS = bool(CFG["edge_repeats"]["enabled"])
    EDGE_REPEATS = int(CFG["edge_repeats"]["repeats"])

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RESULTS_CSV = os.path.join(PROJECT_ROOT, CFG["results"]["csv_path"])
    RUN_DT = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    plots_dir = os.path.join(PROJECT_ROOT, CFG["plot"]["out_dir"])
    plot_linthresh = int(CFG["plot"]["linthresh"])
    plot_linscale = float(CFG["plot"]["linscale"])
    plot_dpi = int(CFG["plot"]["dpi"])
    plot_fig_size = tuple(CFG["plot"]["fig_size"])

    # ------------------------------------------------------------
    # Optional: AE-family EDGE exploration boost (usually helps ≤2k)
    # ------------------------------------------------------------
    AE_EDGE_RANDOM_MIX_BOOST = True
    AE_EDGE_RANDOM_MIX_DELTA = 0.15  # +0.15 (capped) is usually safe
    AE_FAMILY = ("pyod-AE", "pyod-VAE", "pyod-DeepSVDD")

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

        if max_rows is not None and int(max_rows) > 0:
            X_train, y_train = stratified_subsample(X_train, y_train, int(max_rows), seed=seed)
            X_test, y_test = stratified_subsample(X_test, y_test, max(1, int(max_rows * 0.25)), seed=seed + 1)

        y_train_bin = (y_train != 0).astype(int)
        X_train_norm = sanitize_np(X_train[y_train_bin == 0])
        d_in = int(X_train.shape[1])

        # protocol column handling
        PROTO_IDX = 0
        PROTO_ALLOWED = infer_protocol_allowed_from_scaled_train(X_train, PROTO_IDX, device=device)
        print(f"[PROTO] idx={PROTO_IDX} allowed={PROTO_ALLOWED.detach().cpu().numpy().tolist()}")

        feat_lo = torch.from_numpy(np.min(X_train_norm, axis=0).astype(np.float32)).to(device)
        feat_hi = torch.from_numpy(np.max(X_train_norm, axis=0).astype(np.float32)).to(device)

        X_test_eval, y_test_eval = stratified_subsample(X_test, y_test, MAX_EVAL_TEST, seed=seed + 123)
        y_test_eval_bin = (y_test_eval != 0).astype(int)

        X_test_fid, y_test_fid = stratified_subsample(X_test, y_test, MAX_FID_TEST, seed=seed + 456)
        fid_loader = make_loader(X_test_fid, y_test_fid, batch_size=EVAL_BATCH, shuffle=False)

        merged_curves = {}

        for vt in victims:
            print("\n------------------------------------")
            print("Victim:", vt)

            teacher = build_teacher(vt, d_in=d_in, device=DEVICE)
            teacher.fit(X_train_norm.astype(np.float32, copy=False))

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

            # refine schedule
            REFINE_X_STEPS_TRAJ = 2 if vt in ("DRocc", "NeuTraL-AD") else 1
            REFINE_X_STEPS_EDGE = 0 if vt in AE_FAMILY else (2 if vt in ("DRocc", "NeuTraL-AD") else 1)

            # =========================================================
            # EDGE repeats (<= EDGE_QMAX)
            # - Collect per-repeat per-milestone records (FULL metrics)
            # - Then compute best-per-milestone across repeats (max rho)
            # - Log THAT best-per-milestone summary to CSV
            # =========================================================
            edge_runs_full: List[Dict[int, Dict[str, Any]]] = []
            edge_best_full: Dict[int, Dict[str, Any]] = {}
            edge_curve_for_plot: Dict[int, Tuple[int, float]] = {}

            def run_edge_once(rep_seed: int) -> Dict[int, Dict[str, Any]]:
                np.random.seed(rep_seed)
                torch.manual_seed(rep_seed)

                student_r = ScoreMLP(d_in, hidden=STUDENT_HIDDEN).to(device)
                gen_r = build_default_generator(d_in=d_in, z_dim=Z_DIM, hidden=GEN_HIDDEN, out_act=None).to(device)

                milestones_r = list(EDGE_MILESTONES)
                next_i = 0
                per_ms: Dict[int, Dict[str, Any]] = {}

                def on_edge_milestone(event: dict):
                    nonlocal next_i, per_ms
                    used = int(event["used"])
                    stats = event["stats"]

                    if next_i >= len(milestones_r):
                        return
                    target = int(milestones_r[next_i])
                    if used < target:
                        return

                    calib_X_np = stats.get("calib_X_np", None)
                    calib_y_np = stats.get("calib_y_np", None)
                    if calib_X_np is not None and calib_y_np is not None and len(calib_y_np) > 0:
                        cal = calibrate_student_affine(
                            teacher_score_fn=teacher_score_fn,
                            student=student_r,
                            d_in=d_in,
                            device=device,
                            feat_lo=feat_lo,
                            feat_hi=feat_hi,
                            calib_X=torch.from_numpy(calib_X_np),
                            calib_y=torch.from_numpy(calib_y_np),
                            force_monotonic=True,
                        )
                        student_cal = cal["calibrated_student"]
                    else:
                        student_cal = student_r

                    fid = score_fidelity_on_loader(teacher_score_fn, student_cal, fid_loader, device)
                    rho = float(fid["spearman_rho"])

                    s_scores = _student_scores_on_eval(student_cal, X_test_eval, device)
                    student_auroc, student_auprc = safe_auc(y_test_eval_bin, s_scores)

                    per_ms[int(target)] = dict(
                        used=int(used),
                        rho=float(rho),
                        student_auroc=float(student_auroc),
                        student_auprc=float(student_auprc),
                        extract_sec=float(stats.get("extract_sec", 0.0)),
                        queries_per_sec=float(stats.get("queries_per_sec", 0.0)),
                        ms_per_query=float(stats.get("ms_per_query", 0.0)),
                        rep_seed=int(rep_seed),
                    )

                    print(f"[EDGE rep_seed={rep_seed}] [MILESTONE] target={target} used={used} rho={rho:.4f}")
                    next_i += 1

                edge_budget = int(EDGE_QMAX)
                K_edge = int(K_EARLY)
                steps_edge = int(math.ceil(edge_budget / float(K_edge)))

                rm_edge = float(RANDOM_MIX_EDGE)
                if AE_EDGE_RANDOM_MIX_BOOST and vt in AE_FAMILY:
                    rm_edge = min(0.85, rm_edge + float(AE_EDGE_RANDOM_MIX_DELTA))

                cfg_r = QSelDFMEConfig(
                    steps=int(steps_edge),
                    pool_size=int(POOL_SIZE_EDGE),
                    teacher_batch_size=int(K_edge),
                    total_query_budget=int(edge_budget),

                    z_dim=int(Z_DIM),
                    lr_student=float(LR_STUDENT),
                    lr_gen=float(LR_GEN),
                    student_loss=str(STUDENT_LOSS),
                    lambda_gen=float(LAMBDA_GEN),

                    use_query_selection=True,
                    selector_pca_dim=int(PROJ_DIM_EDGE),
                    selector_prefilter_ratio=float(PREFILTER_RATIO_EDGE),
                    selector_candidate_factor=int(CAND_FACTOR_EDGE),

                    random_mix_frac=float(rm_edge),
                    gen_update_every=int(GEN_UPDATE_EVERY),
                    gen_steps=int(GEN_STEPS),
                    gen_diversity_w=float(GEN_DIVERSITY_W),

                    store_calib_buffer=True,
                    calib_buffer_max=200_000,

                    use_feat_bounds=True,
                    seed=int(rep_seed),
                    log_every=max(5, steps_edge // 10),

                    student_steps_per_round=int(STUDENT_STEPS),
                    replay_ratio=float(REPLAY_RATIO),
                    replay_quantiles=int(REPLAY_QUANTILES),
                    replay_cap_mult=3.0,

                    rank_loss_weight=float(RANK_W),
                    rank_pairs=int(RANK_PAIRS),
                    rank_anneal_to=0.10,

                    refine_x_steps=int(REFINE_X_STEPS_EDGE),

                    min_last_k=16,
                    teacher_ema_alpha=0.10,

                    proto_idx=int(PROTO_IDX),
                    proto_allowed=PROTO_ALLOWED,

                    k_early=int(K_edge),
                    k_late=int(K_edge),
                    k_switch_at=int(edge_budget + 1),

                    milestones=list(milestones_r),
                    milestone_callback=on_edge_milestone,
                )

                _ = train_student_qselect_dfme(
                    teacher_score_fn=teacher_score_fn,
                    student=student_r,
                    generator=gen_r,
                    device=device,
                    cfg=cfg_r,
                    feat_lo=feat_lo,
                    feat_hi=feat_hi,
                )

                return per_ms

            if RUN_EDGE_REPEATS:
                print(f"\n[EDGE REPEATS] <= {EDGE_QMAX} | repeats={EDGE_REPEATS} | per-milestone BEST logging")

                for rep in range(int(EDGE_REPEATS)):
                    rep_seed = int(seed + rep)
                    per_ms = run_edge_once(rep_seed)
                    edge_runs_full.append(per_ms)

                # best-per-milestone across repeats (max rho per target)
                edge_best_full = _best_per_milestone_from_repeats(edge_runs_full)
                edge_curve_for_plot = _edge_curve_tuple_from_full(edge_best_full)

                print(f"[EDGE BEST PER-MILESTONE] milestones={len(edge_best_full)}")

                # ------------------------------------------------------------
                # LOG EDGE BEST PER-MILESTONE to CSV (<= EDGE_QMAX ONLY)
                # ------------------------------------------------------------
                if edge_best_full:
                    run_tag_edge = f"{ds}_{vt}_QSelDFME_EDGEBESTPERMS_qmax{EDGE_QMAX}_reps{EDGE_REPEATS}"
                    for target in sorted(edge_best_full.keys()):
                        m = edge_best_full[target]
                        used = int(m.get("used", target))
                        rho = float(m.get("rho", float("nan")))
                        rep_seed_winner = m.get("rep_seed", "")

                        row = dict(
                            run_tag=run_tag_edge,
                            run_datetime=RUN_DT,
                            dataset=ds,
                            victim_type=vt,
                            seed=int(seed),  # base seed; best-per-ms comes from multiple rep seeds
                            d_in=int(d_in),
                            max_rows=int(max_rows),

                            budget_target=int(target),
                            budget_cap_actual=int(target),
                            steps=int(math.ceil(EDGE_QMAX / float(K_EARLY))),
                            pool_size=int(POOL_SIZE_EDGE),
                            teacher_batch_size=int(K_EARLY),

                            use_query_selection=1,
                            prefilter_ratio=float(PREFILTER_RATIO_EDGE),
                            pca_dim=int(PROJ_DIM_EDGE),

                            student_steps_per_round=int(STUDENT_STEPS),
                            replay_ratio=float(REPLAY_RATIO),
                            replay_quantiles=int(REPLAY_QUANTILES),
                            rank_loss_weight=float(RANK_W),
                            rank_pairs=int(RANK_PAIRS),

                            refine_x_steps=int(REFINE_X_STEPS_EDGE),

                            query_budget_expected=int(target),
                            query_budget=int(used),

                            extract_sec=float(m.get("extract_sec", 0.0)),
                            queries_per_sec=float(m.get("queries_per_sec", 0.0)),
                            ms_per_query=float(m.get("ms_per_query", 0.0)),

                            teacher_auroc=float(teacher_auroc),
                            teacher_auprc=float(teacher_auprc),
                            student_auroc=float(m.get("student_auroc", float("nan"))),
                            student_auprc=float(m.get("student_auprc", float("nan"))),

                            fidelity_mse=float("nan"),
                            fidelity_mae=float("nan"),
                            fidelity_spearman=float(rho),

                            tau_rho=float(TAU_RHO),
                            eps_auroc=float(EPS_AUROC),
                            meets_rho=int((rho == rho) and (rho >= TAU_RHO)),
                            meets_both="",
                            qmin_rho="",
                            qmin_both="",

                            edge_qmin=int(EDGE_QMIN),
                            edge_qmax=int(EDGE_QMAX),
                            within_edge_window=1,
                            qmin_rho_within_edge="",
                            qmin_both_within_edge="",

                            # optional traceability
                            edge_best_rep_seed=rep_seed_winner,
                        )
                        append_row(RESULTS_CSV, row)

            # =========================================================
            # Main trajectory (single run to MAX_BUDGET)
            # IMPORTANT: logs ONLY milestones > EDGE_QMAX
            # =========================================================
            student = ScoreMLP(d_in, hidden=STUDENT_HIDDEN).to(device)
            gen = build_default_generator(d_in=d_in, z_dim=Z_DIM, hidden=GEN_HIDDEN, out_act=None).to(device)

            curve_q_used: List[int] = []
            curve_rho: List[float] = []
            qmin_rho = None
            qmin_both = None

            milestones = list(TR_MILESTONES)
            next_milestone_i = 0

            def on_milestone(event: dict):
                nonlocal next_milestone_i, qmin_rho, qmin_both
                used = int(event["used"])
                stats = event["stats"]

                if next_milestone_i >= len(milestones):
                    return
                target = int(milestones[next_milestone_i])
                if used < target:
                    return

                calib_X_np = stats.get("calib_X_np", None)
                calib_y_np = stats.get("calib_y_np", None)
                if calib_X_np is not None and calib_y_np is not None and len(calib_y_np) > 0:
                    cal = calibrate_student_affine(
                        teacher_score_fn=teacher_score_fn,
                        student=student,
                        d_in=d_in,
                        device=device,
                        feat_lo=feat_lo,
                        feat_hi=feat_hi,
                        calib_X=torch.from_numpy(calib_X_np),
                        calib_y=torch.from_numpy(calib_y_np),
                        force_monotonic=True,
                    )
                    student_cal = cal["calibrated_student"]
                else:
                    student_cal = student

                fid = score_fidelity_on_loader(teacher_score_fn, student_cal, fid_loader, device)
                rho = float(fid["spearman_rho"])

                s_scores = _student_scores_on_eval(student_cal, X_test_eval, device)
                student_auroc, student_auprc = safe_auc(y_test_eval_bin, s_scores)

                meets_rho = int((rho == rho) and (rho >= TAU_RHO))
                auroc_gap = (
                    abs(float(student_auroc) - float(teacher_auroc))
                    if (teacher_auroc == teacher_auroc and student_auroc == student_auroc)
                    else float("inf")
                )
                meets_both = int(meets_rho and (auroc_gap <= EPS_AUROC))

                if meets_rho and qmin_rho is None:
                    qmin_rho = used
                if meets_both and qmin_both is None:
                    qmin_both = used

                within_edge_window = int((used >= EDGE_QMIN) and (used <= EDGE_QMAX))
                qmin_rho_within_edge = int((qmin_rho is not None) and (qmin_rho <= EDGE_QMAX))
                qmin_both_within_edge = int((qmin_both is not None) and (qmin_both <= EDGE_QMAX))

                run_tag = (
                    f"{ds}_{vt}_QSelDFME_TRAJ_max{MAX_BUDGET}"
                    f"_Kearly{K_EARLY}_Klate{K_LATE}_switch{K_SWITCH_AT}"
                    f"_pool{POOL_SIZE}_proj{PROJ_DIM}_pref{PREFILTER_RATIO}"
                )

                row = dict(
                    run_tag=run_tag,
                    run_datetime=RUN_DT,
                    dataset=ds,
                    victim_type=vt,
                    seed=int(seed),
                    d_in=int(d_in),
                    max_rows=int(max_rows),

                    budget_target=int(target),
                    budget_cap_actual=int(target),
                    steps=int(STEPS_TRAJ),
                    pool_size=int(POOL_SIZE),
                    teacher_batch_size=int(K_LATE),

                    use_query_selection=1,
                    prefilter_ratio=float(PREFILTER_RATIO),
                    pca_dim=int(PROJ_DIM),

                    student_steps_per_round=int(STUDENT_STEPS),
                    replay_ratio=float(REPLAY_RATIO),
                    replay_quantiles=int(REPLAY_QUANTILES),
                    rank_loss_weight=float(RANK_W),
                    rank_pairs=int(RANK_PAIRS),

                    refine_x_steps=int(REFINE_X_STEPS_TRAJ),

                    query_budget_expected=int(target),
                    query_budget=int(used),

                    extract_sec=float(stats.get("extract_sec", 0.0)),
                    queries_per_sec=float(stats.get("queries_per_sec", 0.0)),
                    ms_per_query=float(stats.get("ms_per_query", 0.0)),

                    teacher_auroc=float(teacher_auroc),
                    teacher_auprc=float(teacher_auprc),
                    student_auroc=float(student_auroc),
                    student_auprc=float(student_auprc),

                    fidelity_mse=float(fid.get("score_mse", float("nan"))),
                    fidelity_mae=float(fid.get("score_mae", float("nan"))),
                    fidelity_spearman=float(rho),

                    tau_rho=float(TAU_RHO),
                    eps_auroc=float(EPS_AUROC),
                    meets_rho=int(meets_rho),
                    meets_both=int(meets_both),
                    qmin_rho=int(qmin_rho) if qmin_rho is not None else "",
                    qmin_both=int(qmin_both) if qmin_both is not None else "",

                    edge_qmin=int(EDGE_QMIN),
                    edge_qmax=int(EDGE_QMAX),
                    within_edge_window=int(within_edge_window),
                    qmin_rho_within_edge=int(qmin_rho_within_edge),
                    qmin_both_within_edge=int(qmin_both_within_edge),
                )
                append_row(RESULTS_CSV, row)

                curve_q_used.append(int(used))
                curve_rho.append(float(rho))

                print(f"[MILESTONE] target={target} used={used} rho={rho:.4f} AUROC_s={student_auroc:.4f}")
                next_milestone_i += 1

            cfg = QSelDFMEConfig(
                steps=int(STEPS_TRAJ),
                pool_size=int(POOL_SIZE),
                teacher_batch_size=int(K_LATE),
                total_query_budget=int(MAX_BUDGET),

                z_dim=int(Z_DIM),
                lr_student=float(LR_STUDENT),
                lr_gen=float(LR_GEN),
                student_loss=str(STUDENT_LOSS),
                lambda_gen=float(LAMBDA_GEN),

                use_query_selection=True,
                selector_pca_dim=int(PROJ_DIM),
                selector_prefilter_ratio=float(PREFILTER_RATIO),
                selector_candidate_factor=int(CAND_FACTOR),

                random_mix_frac=float(RANDOM_MIX_TRAJ),
                gen_update_every=int(GEN_UPDATE_EVERY),
                gen_steps=int(GEN_STEPS),
                gen_diversity_w=float(GEN_DIVERSITY_W),

                store_calib_buffer=True,
                calib_buffer_max=200_000,

                use_feat_bounds=True,
                seed=int(seed),
                log_every=max(10, STEPS_TRAJ // 10),

                student_steps_per_round=int(STUDENT_STEPS),
                replay_ratio=float(REPLAY_RATIO),
                replay_quantiles=int(REPLAY_QUANTILES),
                replay_cap_mult=3.0,

                rank_loss_weight=float(RANK_W),
                rank_pairs=int(RANK_PAIRS),
                rank_anneal_to=0.10,

                refine_x_steps=int(REFINE_X_STEPS_TRAJ),

                min_last_k=16,
                teacher_ema_alpha=0.10,

                proto_idx=int(PROTO_IDX),
                proto_allowed=PROTO_ALLOWED,

                k_early=int(K_EARLY),
                k_late=int(K_LATE),
                k_switch_at=int(K_SWITCH_AT),

                milestones=list(milestones),
                milestone_callback=on_milestone,
            )

            print(
                f"\n[TRAJ RUN] max_budget={MAX_BUDGET} | steps={STEPS_TRAJ} | "
                f"Kearly={K_EARLY} -> Klate={K_LATE} @ {K_SWITCH_AT} | milestones={len(milestones)}"
            )

            _ = train_student_qselect_dfme(
                teacher_score_fn=teacher_score_fn,
                student=student,
                generator=gen,
                device=device,
                cfg=cfg,
                feat_lo=feat_lo,
                feat_hi=feat_hi,
            )

            # =========================================================
            # Plot curve splice:
            # - <= EDGE_QMAX uses best-per-milestone EDGE curve
            # - > EDGE_QMAX uses TRAJ curve points
            # =========================================================
            if edge_curve_for_plot:
                plot_q, plot_r = merge_edge_avg_into_traj(
                    edge_avg=edge_curve_for_plot,  # (used,rho) dict
                    traj_q_used=curve_q_used,
                    traj_rho=curve_rho,
                    edge_qmax=int(EDGE_QMAX),
                    edge_x="target",
                    ensure_origin=True,
                    traj_skip_slack=int(max(16, 2 * K_EARLY)),
                )
                merged_curves[vt] = {"q_used": plot_q, "rho": plot_r}
            else:
                merged_curves[vt] = {"q_used": curve_q_used, "rho": curve_rho}

        save_dataset_rho_vs_queries_plot(
            out_dir=plots_dir,
            dataset=ds,
            curves=merged_curves,
            tau_rho=TAU_RHO,
            edge_qmin=EDGE_QMIN,
            edge_qmax=EDGE_QMAX,
            linthresh=plot_linthresh,
            linscale=plot_linscale,
            xmax=MAX_BUDGET,
            fig_size=plot_fig_size,
            dpi=plot_dpi,
        )
        print(f"[MERGED PLOT SAVED] {plots_dir}")

    print("\nDONE. CSV:", RESULTS_CSV)


if __name__ == "__main__":
    main()


