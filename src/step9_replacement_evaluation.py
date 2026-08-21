import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pair_removal import (
    load_config,
    load_fixed_se_cache,
    set_seed,
)

# Step 9 不重新实现 correspondence replacement。
# 直接固定并复用已经验证过的 Step 8 v2 mechanism。
from step8_correspondence_replacement_v2 import (
    empty_quality,
    fit_cca,
    mutual_matches,
    oracle_pairs,
    pseudo_quality,
    rank_signature,
    replay_weak_ids,
    self_check as step8_self_check,
)


ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUT = ROOT / "results" / "step9"

# Step 9 正式 Pair Sweep。
# 重点加密之前 Step 6 找到的 failure-transition 区域。
DEFAULT_N_REALS = [
    500,
    300,
    200,
    175,
    150,
    125,
    100,
    75,
    50,
    25,
    10,
]


# ============================================================
# Result helper
# ============================================================

def make_row(
    seed,
    condition,
    n_real,
    n_pseudo,
    mutual_pool_size,
    quality,
    recall,
):
    """
    Convert one CCA evaluation into one CSV row.
    """

    t2i = recall["t2i"]
    i2t = recall["i2t"]

    return {
        "seed": int(seed),
        "condition": condition,
        "n_real": int(n_real),
        "n_pseudo": int(n_pseudo),

        # 当前 N_real 下能够生成多少 mutual relation candidates
        "mutual_pool_size": int(mutual_pool_size),

        # Step 8 v2 quality diagnostics
        **quality,

        # CCA retrieval
        "cca_t2i_R1": t2i["R1"],
        "cca_t2i_R5": t2i["R5"],
        "cca_t2i_R10": t2i["R10"],

        "cca_i2t_R1": i2t["R1"],
        "cca_i2t_R5": i2t["R5"],
        "cca_i2t_R10": i2t["R10"],

        "cca_mean_R1": (
            t2i["R1"] + i2t["R1"]
        ) / 2,

        "cca_mean_R5": (
            t2i["R5"] + i2t["R5"]
        ) / 2,

        # Step 9 PRIMARY METRIC
        "cca_mean_R10": (
            t2i["R10"] + i2t["R10"]
        ) / 2,
    }


def stable_rng(seed, n_real, stream):
    """
    Make shuffled/oracle control sampling reproducible.

    Important:
    Control randomness should depend on seed + N_real,
    rather than depending on the order in which the sweep
    happens to be executed.
    """

    mixed = (
        int(seed) * 1_000_003
        + int(n_real) * 10_007
        + int(stream) * 97
        + 17
    ) % (2 ** 32)

    return np.random.default_rng(mixed)


def print_rows(rows):
    """
    Print one N_real result table.
    """

    print(
        f"{'Condition':>16} | "
        f"{'Exact':>7} | "
        f"{'Avail':>7} | "
        f"{'Hit@10':>7} | "
        f"{'Hit@100':>7} | "
        f"{'MedRank':>9} | "
        f"{'Mean@10':>7}"
    )

    print("-" * 91)

    def pct(v):
        if v == "":
            return "-"
        return f"{100 * float(v):.1f}%"

    for r in rows:

        if r["median_gt_rank"] == "":
            med = "-"
        else:
            med = f"{float(r['median_gt_rank']):.1f}"

        print(
            f"{r['condition']:>16} | "
            f"{pct(r['exact_precision']):>7} | "
            f"{pct(r['gt_available_rate']):>7} | "
            f"{pct(r['semantic_hit_R10']):>7} | "
            f"{pct(r['semantic_hit_R100']):>7} | "
            f"{med:>9} | "
            f"{float(r['cca_mean_R10']):>7.2f}"
        )


# ============================================================
# Single-seed Step 9 experiment
# ============================================================

def run_seed(args):

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )

    set_seed(args.seed)

    # --------------------------------------------------------
    # 1. Load config + fixed SE
    # --------------------------------------------------------

    config = load_config(args.data)

    n_components = int(
        config["n_components"]
    )

    cache = load_fixed_se_cache(
        args.data,
        args.seed,
        args.cache,
    )

    train1 = cache["train_se1"]
    train2 = cache["train_se2"]

    test1 = cache["test_se1"]
    test2 = cache["test_se2"]

    pair_order = cache["pair_order"]

    # --------------------------------------------------------
    # 2. Validate pair sweep
    # --------------------------------------------------------

    n_reals = list(
        dict.fromkeys(
            int(n)
            for n in args.n_reals
        )
    )

    for n_real in n_reals:

        if (
            n_real <= n_components
            or
            n_real > len(pair_order)
        ):

            raise ValueError(
                f"n_real={n_real} is invalid; "
                f"expected "
                f"{n_components + 1} <= n_real "
                f"<= {len(pair_order)}."
            )

    if args.n_pseudo <= 0:
        raise ValueError(
            "n_pseudo must be positive."
        )

    # --------------------------------------------------------
    # 3. Keep Step 8 v2 candidate protocol unchanged
    # --------------------------------------------------------
    #
    # VERY IMPORTANT:
    #
    # Even when N_real < 500,
    # the complete 500-pair master block is excluded
    # from pseudo candidates.
    #
    # Therefore:
    #
    #   N_real = 100
    #
    # does NOT allow the remaining 400 known real pairs
    # to leak into pseudo matching.
    #
    # This keeps the Step 9 replacement experiment clean.
    # --------------------------------------------------------

    candidate_idx = np.setdiff1d(
        np.arange(len(train1)),
        pair_order,
        assume_unique=True,
    )

    # --------------------------------------------------------
    # 4. Replay provenance
    # --------------------------------------------------------

    encoded1 = torch.load(
        ROOT
        / "data"
        / args.data
        / "encoded1.pt",

        map_location="cpu",
    )

    n_original_train = (
        len(encoded1)
        -
        int(config["n_test"])
    )

    del encoded1

    ids1, ids2 = replay_weak_ids(
        n_original_train,
        len(pair_order),
        args.seed,
    )

    if (
        len(ids1) != len(train1)
        or
        len(ids2) != len(train2)
    ):

        raise RuntimeError(
            "Weak-data replay != fixed-SE cache. "
            "build_fixed_se.py/data.py may have changed."
        )

    if not np.array_equal(
        ids1[pair_order],
        ids2[pair_order],
    ):

        raise RuntimeError(
            "Master-pair provenance replay failed."
        )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("=" * 72)
    print(
        "Step 9 - Replacement Evaluation"
    )
    print("=" * 72)

    print(
        f"Dataset              : "
        f"{args.data}"
    )

    print(
        f"Seed                 : "
        f"{args.seed}"
    )

    print(
        f"CCA components       : "
        f"{n_components}"
    )

    print(
        f"Master real-pair pool: "
        f"{len(pair_order)}"
    )

    print(
        f"Pseudo candidates    : "
        f"{len(candidate_idx)}"
    )

    print(
        f"Fixed pseudo budget  : "
        f"{args.n_pseudo}"
    )

    print(
        f"Real-pair sweep      : "
        f"{n_reals}"
    )

    print(
        "Provenance replay    : PASS"
    )

    all_rows = []

    # ========================================================
    # 5. Real-pair sweep
    # ========================================================

    for n_real in n_reals:

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"N_real = {n_real}"
        )

        print("=" * 72)

        # ----------------------------------------------------
        # Nested real-pair protocol
        #
        # P10 ⊂ P25 ⊂ ... ⊂ P500
        # ----------------------------------------------------

        real_idx = pair_order[
            :n_real
        ].copy()

        # ----------------------------------------------------
        # Step 8 relation mechanism
        #
        # Relation signatures MUST be regenerated for every
        # N_real because the signature itself is defined
        # relative to the currently available real anchors.
        # ----------------------------------------------------

        image_sig = rank_signature(
            train1[candidate_idx],
            train1[real_idx],
        )

        text_sig = rank_signature(
            train2[candidate_idx],
            train2[real_idx],
        )

        (
            image_local,
            text_local,
            confidence,
        ) = mutual_matches(
            image_sig,
            text_sig,
        )

        pseudo_image = candidate_idx[
            image_local
        ]

        pseudo_text = candidate_idx[
            text_local
        ]

        mutual_pool_size = len(
            pseudo_image
        )

        if (
            mutual_pool_size
            <
            args.n_pseudo
        ):

            raise RuntimeError(
                f"N_real={n_real}: "
                f"only {mutual_pool_size} "
                f"mutual relation pairs available, "
                f"but n_pseudo={args.n_pseudo}."
            )

        # Fixed pseudo budget:
        #
        # always take top 75 by Step 8 confidence
        # unless --n_pseudo is explicitly changed.

        pi = pseudo_image[
            :args.n_pseudo
        ]

        pt = pseudo_text[
            :args.n_pseudo
        ]

        rows = []

        # ====================================================
        # Condition 1
        #
        # REAL ONLY
        # ====================================================

        recall = fit_cca(
            train1,
            train2,
            test1,
            test2,

            real_idx,
            real_idx,

            n_components,
        )

        rows.append(
            make_row(
                args.seed,

                "real_only",

                n_real,
                0,

                mutual_pool_size,

                empty_quality(),

                recall,
            )
        )

        # ====================================================
        # Condition 2
        #
        # RELATION REPLACEMENT
        #
        # N_real real pairs
        # +
        # fixed 75 relation constraints
        # ====================================================

        relation_quality = pseudo_quality(
            pi,
            pt,

            candidate_idx,

            ids1,
            ids2,

            train1,
            train2,
        )

        recall = fit_cca(
            train1,
            train2,
            test1,
            test2,

            np.r_[
                real_idx,
                pi,
            ],

            np.r_[
                real_idx,
                pt,
            ],

            n_components,
        )

        rows.append(
            make_row(
                args.seed,

                f"relation_{args.n_pseudo}",

                n_real,
                args.n_pseudo,

                mutual_pool_size,

                relation_quality,

                recall,
            )
        )

        # ====================================================
        # Condition 3
        #
        # SHUFFLED NEGATIVE CONTROL
        #
        # Keep exactly the same relation-selected image/text
        # samples but destroy their cross-modal assignment.
        # ====================================================

        shuffle_rng = stable_rng(
            args.seed,
            n_real,
            stream=1,
        )

        shuffled_pt = (
            shuffle_rng.permutation(
                pt
            )
        )

        shuffled_quality = pseudo_quality(
            pi,
            shuffled_pt,

            candidate_idx,

            ids1,
            ids2,

            train1,
            train2,
        )

        recall = fit_cca(
            train1,
            train2,
            test1,
            test2,

            np.r_[
                real_idx,
                pi,
            ],

            np.r_[
                real_idx,
                shuffled_pt,
            ],

            n_components,
        )

        rows.append(
            make_row(
                args.seed,

                f"shuffled_{args.n_pseudo}",

                n_real,
                args.n_pseudo,

                mutual_pool_size,

                shuffled_quality,

                recall,
            )
        )

        # ====================================================
        # Condition 4
        #
        # ORACLE TRUE-PAIR CONTROL
        #
        # Same extra correspondence budget,
        # but all correspondences are true.
        # ====================================================

        oracle_rng = stable_rng(
            args.seed,
            n_real,
            stream=2,
        )

        oi, ot = oracle_pairs(
            candidate_idx,

            ids1,
            ids2,

            args.n_pseudo,

            oracle_rng,
        )

        oracle_quality = pseudo_quality(
            oi,
            ot,

            candidate_idx,

            ids1,
            ids2,

            train1,
            train2,
        )

        recall = fit_cca(
            train1,
            train2,
            test1,
            test2,

            np.r_[
                real_idx,
                oi,
            ],

            np.r_[
                real_idx,
                ot,
            ],

            n_components,
        )

        rows.append(
            make_row(
                args.seed,

                f"oracle_{args.n_pseudo}",

                n_real,
                args.n_pseudo,

                mutual_pool_size,

                oracle_quality,

                recall,
            )
        )

        # ----------------------------------------------------
        # Save in memory
        # ----------------------------------------------------

        all_rows.extend(
            rows
        )

        # ----------------------------------------------------
        # Print this N_real result
        # ----------------------------------------------------

        print(
            f"Mutual pseudo pool   : "
            f"{mutual_pool_size}"
        )

        print(
            f"Best mutual confidence: "
            f"{confidence[0]:.6f}"
        )

        print_rows(
            rows
        )

    # ========================================================
    # 6. Save one seed
    # ========================================================

    out = (
        Path(args.output_dir)
        / args.data
        / f"seed{args.seed}.csv"
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        all_rows
    ).to_csv(
        out,
        index=False,
    )

    print(
        f"\nSaved per-seed results: "
        f"{out}"
    )


# ============================================================
# Pair Equivalent Number
# ============================================================

def inverse_baseline_pairs(
    x,
    y,
    target,
):
    """
    Estimate how many REAL baseline pairs are needed to
    achieve the target performance.

    Because a 3-seed empirical baseline curve may contain
    small non-monotonic fluctuations, first construct a
    monotonic envelope and then linearly interpolate.

    This is used ONLY as a descriptive Pair Equivalent
    Number metric. Raw curves are still saved separately.
    """

    order = np.argsort(
        x
    )

    x = np.asarray(
        x,
        dtype=np.float64,
    )[order]

    y = np.asarray(
        y,
        dtype=np.float64,
    )[order]

    # Expected Pair-Performance relation should generally
    # improve as more real pairs are supplied.
    #
    # Remove tiny empirical reversals caused by seed noise.

    y = np.maximum.accumulate(
        y
    )

    # Relation performance outside measured baseline range:
    # do NOT extrapolate.

    if target < y[0]:

        return (
            np.nan,
            "below_baseline_range",
        )

    if target > y[-1]:

        return (
            np.nan,
            "above_baseline_range",
        )

    # Remove plateaus before inverse interpolation.

    keep = np.r_[
        True,
        np.diff(y) > 1e-12,
    ]

    x_unique = x[
        keep
    ]

    y_unique = y[
        keep
    ]

    if len(y_unique) == 1:

        return (
            float(x_unique[0]),
            "flat_baseline",
        )

    equivalent = np.interp(
        target,
        y_unique,
        x_unique,
    )

    return (
        float(equivalent),
        "interpolated",
    )


# ============================================================
# 3-seed aggregation
# ============================================================

def summarize(args):

    base = (
        Path(args.output_dir)
        / args.data
    )

    frames = []

    # --------------------------------------------------------
    # 1. Load three seeds
    # --------------------------------------------------------

    for seed in args.seeds:

        path = (
            base
            / f"seed{seed}.csv"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing {path}. "
                f"Run seed {seed} "
                f"before --summarize."
            )

        frames.append(
            pd.read_csv(
                path
            )
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 2. Metrics to aggregate
    # --------------------------------------------------------

    metric_cols = [
        "mutual_pool_size",

        "exact_precision",
        "gt_available_rate",
        "semantic_hit_R10",
        "semantic_hit_R100",
        "median_gt_rank",

        "cca_t2i_R1",
        "cca_t2i_R5",
        "cca_t2i_R10",

        "cca_i2t_R1",
        "cca_i2t_R5",
        "cca_i2t_R10",

        "cca_mean_R1",
        "cca_mean_R5",
        "cca_mean_R10",
    ]

    # --------------------------------------------------------
    # 3. Mean / std over seeds
    # --------------------------------------------------------

    summary = (
        df.groupby(
            [
                "n_real",
                "condition",
                "n_pseudo",
            ],
            as_index=False,
        )[metric_cols]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    # Flatten pandas MultiIndex column names.

    summary.columns = [
        "_".join(
            [
                str(x)
                for x in col
                if str(x) != ""
            ]
        ).rstrip("_")
        if isinstance(col, tuple)
        else col

        for col in summary.columns
    ]

    summary["n_seeds"] = len(
        args.seeds
    )

    summary_path = (
        base
        / "summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # 4. Pair Equivalent Number
    # ========================================================

    baseline = (
        summary[
            summary["condition"]
            ==
            "real_only"
        ]
        .sort_values(
            "n_real"
        )
    )

    relation = (
        summary[
            summary["condition"]
            ==
            f"relation_{args.n_pseudo}"
        ]
        .sort_values(
            "n_real"
        )
    )

    baseline_map = dict(
        zip(
            baseline[
                "n_real"
            ].astype(int),

            baseline[
                "cca_mean_R10_mean"
            ],
        )
    )

    bx = baseline[
        "n_real"
    ].to_numpy(
        dtype=float
    )

    by = baseline[
        "cca_mean_R10_mean"
    ].to_numpy(
        dtype=float
    )

    pen_rows = []

    for _, r in relation.iterrows():

        n_real = int(
            r["n_real"]
        )

        relation_perf = float(
            r[
                "cca_mean_R10_mean"
            ]
        )

        baseline_perf = float(
            baseline_map[
                n_real
            ]
        )

        (
            equivalent_pairs,
            status,
        ) = inverse_baseline_pairs(
            bx,
            by,
            relation_perf,
        )

        if np.isfinite(
            equivalent_pairs
        ):

            pen = (
                equivalent_pairs
                -
                n_real
            )

        else:

            pen = np.nan

        pen_rows.append(
            {
                "n_real":
                    n_real,

                "n_pseudo":
                    args.n_pseudo,

                "baseline_mean_R10":
                    baseline_perf,

                "relation_mean_R10":
                    relation_perf,

                "replacement_gain_R10":
                    relation_perf
                    -
                    baseline_perf,

                "equivalent_baseline_pairs":
                    equivalent_pairs,

                "pair_equivalent_number":
                    pen,

                "pen_status":
                    status,
            }
        )

    pen_df = pd.DataFrame(
        pen_rows
    ).sort_values(
        "n_real"
    )

    pen_path = (
        base
        / "pair_equivalent_number.csv"
    )

    pen_df.to_csv(
        pen_path,
        index=False,
    )

    # ========================================================
    # 5. Pair-Performance Curve
    # ========================================================

    conditions = [
        "real_only",

        f"relation_{args.n_pseudo}",

        f"shuffled_{args.n_pseudo}",

        f"oracle_{args.n_pseudo}",
    ]

    labels = {
        "real_only":
            "Real only",

        f"relation_{args.n_pseudo}":
            f"Relation +{args.n_pseudo}",

        f"shuffled_{args.n_pseudo}":
            f"Shuffled +{args.n_pseudo}",

        f"oracle_{args.n_pseudo}":
            f"Oracle +{args.n_pseudo}",
    }

    markers = [
        "o",
        "s",
        "^",
        "D",
    ]

    plt.figure(
        figsize=(
            9,
            5.5,
        )
    )

    for (
        condition,
        marker,
    ) in zip(
        conditions,
        markers,
    ):

        part = (
            summary[
                summary["condition"]
                ==
                condition
            ]
            .sort_values(
                "n_real"
            )
        )

        if part.empty:
            continue

        plt.errorbar(
            part[
                "n_real"
            ],

            part[
                "cca_mean_R10_mean"
            ],

            yerr=part[
                "cca_mean_R10_std"
            ].fillna(
                0.0
            ),

            marker=marker,

            capsize=3,

            linewidth=1.6,

            label=labels[
                condition
            ],
        )

    ticks = sorted(
        summary[
            "n_real"
        ]
        .astype(int)
        .unique()
    )

    plt.xticks(
        ticks,
        rotation=45,
    )

    plt.xlabel(
        "Number of real pairs"
    )

    plt.ylabel(
        "CCA Mean Recall@10"
    )

    plt.title(
        "Step 9 - Pair-Performance Curve"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    curve_png = (
        base
        / "pair_performance_curve.png"
    )

    curve_pdf = (
        base
        / "pair_performance_curve.pdf"
    )

    plt.savefig(
        curve_png,
        dpi=200,
    )

    plt.savefig(
        curve_pdf
    )

    plt.close()

    # ========================================================
    # 6. Terminal summary
    # ========================================================

    print("=" * 72)

    print(
        "Step 9 summary"
    )

    print("=" * 72)

    print(
        f"Seeds    : "
        f"{args.seeds}"
    )

    print(
        f"Summary  : "
        f"{summary_path}"
    )

    print(
        f"PEN      : "
        f"{pen_path}"
    )

    print(
        f"Curve PNG: "
        f"{curve_png}"
    )

    print(
        f"Curve PDF: "
        f"{curve_pdf}"
    )

    print()

    print(
        f"{'N_real':>7} | "
        f"{'Baseline':>9} | "
        f"{'Relation':>9} | "
        f"{'Gain':>8} | "
        f"{'EqPairs':>9} | "
        f"{'PEN':>8}"
    )

    print("-" * 67)

    for _, r in pen_df.iterrows():

        equivalent = (
            r[
                "equivalent_baseline_pairs"
            ]
        )

        pen = (
            r[
                "pair_equivalent_number"
            ]
        )

        if pd.isna(
            equivalent
        ):
            eq_text = "-"
        else:
            eq_text = (
                f"{equivalent:.1f}"
            )

        if pd.isna(
            pen
        ):
            pen_text = "-"
        else:
            pen_text = (
                f"{pen:+.1f}"
            )

        print(
            f"{int(r['n_real']):>7} | "
            f"{r['baseline_mean_R10']:>9.2f} | "
            f"{r['relation_mean_R10']:>9.2f} | "
            f"{r['replacement_gain_R10']:>+8.2f} | "
            f"{eq_text:>9} | "
            f"{pen_text:>8}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 9: replacement evaluation "
            "over a real-pair sweep."
        )
    )

    parser.add_argument(
        "data",
        nargs="?",
        default="flickr30",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--n_reals",
        type=int,
        nargs="+",
        default=DEFAULT_N_REALS,
    )

    parser.add_argument(
        "--n_pseudo",
        type=int,
        default=75,
        help=(
            "Fixed relation/shuffled/oracle "
            "extra-pair budget."
        ),
    )

    parser.add_argument(
        "--cache",
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        default=str(
            DEFAULT_OUT
        ),
    )

    parser.add_argument(
        "--self_check",
        action="store_true",
    )

    parser.add_argument(
        "--summarize",
        action="store_true",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            0,
            1,
            2,
        ],
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Self-check
    # --------------------------------------------------------

    if args.self_check:

        step8_self_check()

        print(
            "Step9 import/self-check: PASS"
        )

        return

    # --------------------------------------------------------
    # Aggregate existing seeds
    # --------------------------------------------------------

    if args.summarize:

        summarize(
            args
        )

        return

    # --------------------------------------------------------
    # Run one seed
    # --------------------------------------------------------

    run_seed(
        args
    )


if __name__ == "__main__":
    main()