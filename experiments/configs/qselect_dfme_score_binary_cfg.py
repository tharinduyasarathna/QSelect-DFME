# ============================================================
# FILE: experiments/configs/qselect_dfme_score_binary_cfg.py


CFG = dict(
    seed=42,
    max_rows=200_000,

    datasets=[
        "CIC-IDS2017",
        "InSDN_DatasetCSV",
        "SDN-IoT",
        "UNSW-NB15",
        "CICIoT2023",
        "NSL-KDD",
    ],

    victims=[
        "pyod-AE",
        "pyod-VAE",
        "pyod-DeepSVDD",
        "DRocc",
        "NeuTraL-AD",
    ],

    thresholds=dict(
        tau_rho=0.75,
        eps_auroc=0.02,
    ),

    milestones=dict(
        edge_qmin=50,
        edge_qmax=2_000,
        edge_budgets=[50, 100, 200, 500, 1_000, 1_500, 2_000],

        run_long_tail=True,
        long_tail=[10_000, 20_000, 50_000, 100_000, 500_000, 1_000_000],
    ),

    eval=dict(
        batch=1024,
        max_eval_test=10_000,
        max_fid_test=20_000,
    ),

    # DEFAULT / LONG-TAIL (>=10k)
    selection_default=dict(
        pool_size=8000,
        proj_dim=16,
        prefilter_ratio=0.30,
        cand_factor=4,
        random_mix=0.20,
    ),

    # EDGE (<=2k): keep big pool + low proj_dim, but reduce random a bit for AE stability
    selection_edge=dict(
        pool_size=32_000,
        proj_dim=8,
        prefilter_ratio=0.20,
        cand_factor=10,

        # was 0.60; AE usually improves when this is slightly lower
        random_mix=0.35,
    ),

    # Student: stronger rank + more replay = better Spearman under very low budgets
    student=dict(
        hidden=(512, 512),
        loss="mse_zscore",

        # slightly lower LR helps stability; keep steps_per_round high
        lr=3e-4,
        steps_per_round=12,

        # was 10/11; increase coverage across teacher score spectrum
        replay_ratio=14.0,
        replay_quantiles=13,

        # was 0.85 + 16384; stronger rank tends to help AE/VAE early
        rank_w=1.20,
        rank_pairs=32_768,
    ),

    generator=dict(
        z_dim=64,
        hidden=(512, 512),
        lr=2e-4,
        lambda_gen=0.2,
        update_every=0,
    ),

    schedule=dict(
        steps_traj=1600,

        # was 12; slightly higher early K reduces noisy steps and helps rho <=2k
        k_early=16,

        k_late=2048,
        k_switch_at=5_000,
    ),

    edge_repeats=dict(
        enabled=True,
        repeats=3,
        agg_mode="best",
    ),

    plot=dict(
        out_dir="results/qdfmeplots",
        linthresh=500,
        linscale=1.2,
        dpi=300,
        fig_size=(12.5, 5.8),
    ),

    results=dict(
        csv_path="results/qselect_dfme_score_binary.csv",
    ),
)