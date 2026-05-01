# ============================================================
# FILE: experiments/configs/tempest_score_binary_cfg.py
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

    # milestone budgets (checkpoints) — edge + long tail
    milestones=dict(
        edge_qmin=50,
        edge_qmax=5_000,
        edge_budgets=[50, 100, 200, 500, 1_000, 2_000 ],
        run_long_tail=True,
        long_tail=[5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000],
    ),

    tempest=dict(
        gen_mode="gen_var",       # "gen_var" | "gen_min"
        adv_norm="standard",      # "none" | "standard" | "minmax" 
        epochs=20,
        batch_size=1024,
        lr=1e-3,
        weight_decay=0.0,
        var_eps=1e-6,
        clip_std=6.0,
        teacher_query_chunk=65536,  # only for safe chunking (no algo change)
    ),

    eval=dict(
        batch=512,
    ),

    plot=dict(
        out_dir="results/tempestplots",
        linthresh=500,
        linscale=1.2,
        dpi=250,
        fig_size=(12.5, 5.8),
        tau_rho=0.80,   # for horizontal guideline in plots only
    ),

    results=dict(
        csv_path="results/tempest_score_binary.csv",
    ),
)