# ============================================================
# FILE: experiments/configs/tabextractor_score_binary_cfg.py
# ============================================================

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

    milestones=dict(
        edge_qmin=50,
        edge_qmax=5_000,
        budgets=[50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000,
                 50_000, 100_000, 500_000, 1_000_000],
    ),

    eval=dict(
        batch=512,
        eval_chunk=2048,
        max_eval_test=10_000,
        max_fid_test=20_000,
    ),

    tabextractor=dict(
        train_bs=128,          # attacker batch (queries per iteration)
        max_queries=1_000_000, # ONE run per victim
        log_every_frac=0.10,   # log every ~10% iterations
        label_threshold_mode="median",
        use_feat_bounds=True,
        use_compile=False,
    ),

    plot=dict(
        out_dir="results/TABEXTRACTORplots",
        linthresh=500,
        linscale=1.2,
        dpi=250,
        fig_size=(12.5, 5.8),
        tau_rho=0.80,
    ),

    results=dict(
        csv_path="results/tabextractor_score_binary.csv",
    ),
)