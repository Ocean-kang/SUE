"""
SUE Step 7 - Pair Function Diagnosis.

New-file-only experiment:
- identity: correct vs shuffled correspondence
- diversity: random vs diverse vs redundant correct anchors
- noise: gradually corrupt correspondence identity

CCA-only by design: Step 5/6 already showed MMD is a small refinement stage.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import CCA

from pair_removal import (
    PROJECT_ROOT,
    compute_bidirectional_recall,
    l2_normalize,
    load_config,
    load_fixed_se_cache,
    print_recall,
    save_results_csv,
    set_seed,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "step7_pair_function"


def fit_cca_pairing(train1, train2, test1, test2, x_idx, y_idx, n_components):
    """Same CCA projection convention as pair_removal.py, but X/Y indices may differ."""
    x_idx = np.asarray(x_idx, dtype=np.int64)
    y_idx = np.asarray(y_idx, dtype=np.int64)

    if x_idx.shape != y_idx.shape:
        raise ValueError("x_idx and y_idx must have the same shape.")
    if len(x_idx) <= n_components:
        raise ValueError(
            f"Need > {n_components} pairs for CCA, got {len(x_idx)}."
        )

    cca = CCA(n_components=n_components)
    cca.fit(train1[x_idx], train2[y_idx])

    p1 = cca.x_rotations_
    p2 = cca.y_rotations_
    return test1 @ p1, test2 @ p2


def shifted_pairing(pair_idx):
    """Same samples, zero correct identities."""
    pair_idx = np.asarray(pair_idx, dtype=np.int64)
    if len(pair_idx) < 2:
        raise ValueError("Need at least 2 pairs.")
    # ponytail: pair_order is already randomized; one cyclic shift is a
    # deterministic derangement with no extra shuffle machinery.
    return np.roll(pair_idx, 1)


def select_diverse(features, pool_idx, n_select):
    """Greedy farthest-point sampling in image-SE cosine space."""
    pool_idx = np.asarray(pool_idx, dtype=np.int64)
    x = l2_normalize(features[pool_idx])

    selected = [0]
    min_dist = 1.0 - x @ x[0]
    min_dist[0] = -np.inf

    for _ in range(1, n_select):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, 1.0 - x @ x[nxt])
        min_dist[selected] = -np.inf

    return pool_idx[np.asarray(selected, dtype=np.int64)]


def select_redundant(features, pool_idx, n_select):
    """Select samples nearest to one image-SE anchor."""
    pool_idx = np.asarray(pool_idx, dtype=np.int64)
    x = l2_normalize(features[pool_idx])
    return pool_idx[np.argsort(-(x @ x[0]))[:n_select]]


def noisy_pairing(pair_idx, corruption_order, ratio):
    """Corrupt a nested subset of Y identities; pair count stays fixed."""
    pair_idx = np.asarray(pair_idx, dtype=np.int64)
    y_idx = pair_idx.copy()

    n_corrupt = int(round(ratio * len(pair_idx)))
    if ratio > 0 and n_corrupt < 2:
        n_corrupt = 2
    if n_corrupt == 0:
        return y_idx

    pos = np.asarray(corruption_order[:n_corrupt], dtype=np.int64)
    y_idx[pos] = np.roll(pair_idx[pos], 1)
    return y_idx


def pair_spread(features, indices):
    """Mean pairwise cosine distance: manipulation check for diversity."""
    x = l2_normalize(features[np.asarray(indices, dtype=np.int64)])
    if len(x) < 2:
        return 0.0
    sim = x @ x.T
    upper = np.triu_indices(len(x), k=1)
    return float(np.mean(1.0 - sim[upper]))


def evaluate(args, cache, n_components, experiment, condition, x_idx, y_idx,
             requested_noise=""):
    test1, test2 = fit_cca_pairing(
        cache["train_se1"],
        cache["train_se2"],
        cache["test_se1"],
        cache["test_se2"],
        x_idx,
        y_idx,
        n_components,
    )

    recall = compute_bidirectional_recall(test1, test2)
    print_recall(
        f"[Step7 | {experiment} | {condition} | {len(x_idx)} pairs] CCA-only",
        recall,
    )

    wrong = int(np.count_nonzero(np.asarray(x_idx) != np.asarray(y_idx)))
    n_pairs = len(x_idx)

    return {
        "dataset": args.data,
        "seed": args.seed,
        "experiment": experiment,
        "condition": condition,
        "n_pairs": n_pairs,
        "wrong_pairs": wrong,
        "requested_noise_ratio": requested_noise,
        "actual_noise_ratio": wrong / n_pairs,
        "image_pair_spread": pair_spread(cache["train_se1"], x_idx),
        "text_pair_spread": pair_spread(cache["train_se2"], y_idx),
        "cca_t2i_R1": recall["t2i"]["R1"],
        "cca_t2i_R5": recall["t2i"]["R5"],
        "cca_t2i_R10": recall["t2i"]["R10"],
        "cca_i2t_R1": recall["i2t"]["R1"],
        "cca_i2t_R5": recall["i2t"]["R5"],
        "cca_i2t_R10": recall["i2t"]["R10"],
        "cca_mean_R10": (recall["t2i"]["R10"] + recall["i2t"]["R10"]) / 2,
    }


def self_check():
    idx = np.arange(10, dtype=np.int64)

    shifted = shifted_pairing(idx)
    assert set(shifted) == set(idx)
    assert np.all(shifted != idx)

    noisy = noisy_pairing(idx, np.arange(10), 0.3)
    assert np.count_nonzero(noisy != idx) == 3

    features = np.eye(10)
    assert len(np.unique(select_diverse(features, idx, 5))) == 5
    assert len(np.unique(select_redundant(features, idx, 5))) == 5

    print("Step 7 self-check passed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SUE Step 7 - Pair Function Diagnosis"
    )
    parser.add_argument("data", nargs="?", help="e.g. flickr30")
    parser.add_argument(
        "--experiment",
        choices=("identity", "diversity", "noise"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs", type=int, nargs="+", default=None)
    parser.add_argument(
        "--noise",
        type=float,
        nargs="+",
        default=(0.0, 0.1, 0.2, 0.4, 1.0),
    )
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        return args
    if not args.data:
        parser.error("data is required unless --self-check is used.")
    if not args.experiment:
        parser.error("--experiment is required.")
    if any(r < 0 or r > 1 for r in args.noise):
        parser.error("--noise values must be in [0, 1].")

    if args.pairs is None:
        args.pairs = {
            "identity": [175, 150, 100],
            "diversity": [150],
            "noise": [175],
        }[args.experiment]

    return args


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()

    if args.self_check:
        self_check()
        return

    set_seed(args.seed)
    n_components = int(load_config(args.data)["n_components"])
    cache = load_fixed_se_cache(args.data, args.seed, args.cache)
    pair_order = cache["pair_order"]

    for n_pairs in args.pairs:
        if n_pairs <= n_components:
            raise ValueError(
                f"n_pairs={n_pairs} must be > n_components={n_components}."
            )
        if n_pairs > len(pair_order):
            raise ValueError(
                f"n_pairs={n_pairs}, but pair pool has {len(pair_order)}."
            )

    print("=" * 70)
    print("SUE Step 7 - Pair Function Diagnosis")
    print(f"dataset={args.data} seed={args.seed} experiment={args.experiment}")
    print(f"pairs={args.pairs} n_components={n_components}")
    print("=" * 70)

    rows = []

    if args.experiment == "identity":
        for n_pairs in args.pairs:
            idx = pair_order[:n_pairs].copy()
            rows.append(
                evaluate(
                    args, cache, n_components,
                    "identity", "correct", idx, idx,
                )
            )
            rows.append(
                evaluate(
                    args, cache, n_components,
                    "identity", "shuffled", idx, shifted_pairing(idx),
                )
            )

    elif args.experiment == "diversity":
        for n_pairs in args.pairs:
            selections = {
                "random": pair_order[:n_pairs].copy(),
                "diverse": select_diverse(
                    cache["train_se1"], pair_order, n_pairs
                ),
                "redundant": select_redundant(
                    cache["train_se1"], pair_order, n_pairs
                ),
            }
            for condition, idx in selections.items():
                rows.append(
                    evaluate(
                        args, cache, n_components,
                        "diversity", condition, idx, idx,
                    )
                )

    else:  # noise
        for n_pairs in args.pairs:
            idx = pair_order[:n_pairs].copy()
            corruption_order = np.random.default_rng(
                args.seed
            ).permutation(n_pairs)

            for ratio in args.noise:
                rows.append(
                    evaluate(
                        args, cache, n_components,
                        "noise", f"noise_{ratio:g}",
                        idx,
                        noisy_pairing(idx, corruption_order, ratio),
                        requested_noise=ratio,
                    )
                )

    output_path = (
        Path(args.output_dir)
        / args.data
        / f"seed{args.seed}"
        / f"{args.experiment}.csv"
    )
    save_results_csv(rows, output_path)

    print()
    print(f"{'Condition':>16} | {'Pairs':>5} | {'Wrong':>5} | "
          f"{'T2I@10':>8} | {'I2T@10':>8} | {'Mean@10':>8}")
    print("-" * 72)
    for row in rows:
        print(
            f"{row['condition']:>16} | {row['n_pairs']:>5} | "
            f"{row['wrong_pairs']:>5} | {row['cca_t2i_R10']:>8.2f} | "
            f"{row['cca_i2t_R10']:>8.2f} | {row['cca_mean_R10']:>8.2f}"
        )


if __name__ == "__main__":
    main()
