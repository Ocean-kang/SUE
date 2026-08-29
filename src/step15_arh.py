"""SUE Step15 - ARH information ladder.

Fixed SE -> ARH pseudo-pairs -> CCA -> optional unchanged SUE MMD.
ARH ladder: topk -> rank -> weighted -> soft.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import NearestNeighbors

from pair_removal import (
    compute_bidirectional_recall,
    l2_normalize,
    load_config,
    load_fixed_se_cache,
    print_recall,
    run_mmd,
    save_results_csv,
    set_seed,
)
from step8_correspondence_replacement_v2 import (
    oracle_pairs,
    pseudo_quality,
    replay_weak_ids,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "step15"
DEFAULT_PAIRS = [500, 300, 200, 175, 150, 125, 100, 75, 50, 25, 10]
METHODS = ("topk", "rank", "weighted", "soft")


def anchor_profile(x, anchors, method, k, rank_tau, soft_tau):
    """Represent each sample by its relation to shared paired anchors."""
    sim = l2_normalize(x) @ l2_normalize(anchors).T
    k = min(k, sim.shape[1])
    top = np.argsort(-sim, axis=1)[:, :k]
    rows = np.arange(len(x))[:, None]
    out = np.zeros_like(sim, dtype=np.float32)

    if method == "topk":
        value = np.ones((len(x), k), dtype=np.float32)
    elif method == "rank":
        value = np.broadcast_to(
            np.arange(k, 0, -1, dtype=np.float32), (len(x), k)
        )
    elif method == "weighted":
        w = np.exp(-np.arange(k, dtype=np.float32) / rank_tau)
        value = np.broadcast_to(w, (len(x), k))
    elif method == "soft":
        logits = sim[rows, top] / soft_tau
        logits -= logits.max(axis=1, keepdims=True)
        value = np.exp(logits)
        value /= value.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown ARH method: {method}")

    out[rows, top] = value
    return l2_normalize(out)


def mutual_matches(image_profile, text_profile):
    """Keep only bidirectional nearest matches; margin is confidence."""
    text_nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(text_profile)
    dists, text_best2 = text_nn.kneighbors(image_profile)
    image_nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(image_profile)
    _, image_best = image_nn.kneighbors(text_profile)

    best_text = text_best2[:, 0]
    image_local = np.arange(len(image_profile))
    keep = image_best[best_text, 0] == image_local
    confidence = (dists[:, 1] - dists[:, 0])[keep]

    image_local = image_local[keep]
    text_local = best_text[keep]
    order = np.argsort(-confidence)
    return image_local[order], text_local[order], confidence[order]


def fit_cross_cca(train1, train2, idx1, idx2, n_components):
    """CCA with real same-index pairs plus cross-index ARH pseudo-pairs."""
    if len(idx1) != len(idx2) or len(idx1) <= n_components:
        raise ValueError("Invalid CCA pair indices.")
    cca = CCA(n_components=n_components)
    cca.fit(train1[idx1], train2[idx2])
    return cca.x_rotations_.astype(np.float32), cca.y_rotations_.astype(np.float32)


def project(train1, train2, test1, test2, w1, w2):
    return {
        "train1": train1 @ w1,
        "train2": train2 @ w2,
        "test1": test1 @ w1,
        "test2": test2 @ w2,
    }


def empty_quality():
    nan = float("nan")
    return {
        "exact_precision": nan,
        "gt_available_rate": nan,
        "semantic_hit_R10": nan,
        "semantic_hit_R100": nan,
        "median_gt_rank": nan,
    }


def metric_row(
    dataset,
    seed,
    n_real,
    n_pseudo,
    method,
    arh_k,
    pool_size,
    coverage,
    mean_confidence,
    quality,
    recall,
    final,
):
    nan = float("nan")
    return {
        "dataset": dataset,
        "seed": seed,
        "n_real": n_real,
        "n_pseudo": n_pseudo,
        "method": method,
        "arh_k": arh_k,
        "pseudo_pool_size": pool_size,
        "pseudo_coverage": coverage,
        "mean_confidence": mean_confidence,
        **quality,
        "cca_t2i_R1": recall["t2i"]["R1"],
        "cca_t2i_R5": recall["t2i"]["R5"],
        "cca_t2i_R10": recall["t2i"]["R10"],
        "cca_i2t_R1": recall["i2t"]["R1"],
        "cca_i2t_R5": recall["i2t"]["R5"],
        "cca_i2t_R10": recall["i2t"]["R10"],
        "cca_mean_R10": 0.5 * (recall["t2i"]["R10"] + recall["i2t"]["R10"]),
        "final_t2i_R1": final["t2i"]["R1"] if final else nan,
        "final_t2i_R5": final["t2i"]["R5"] if final else nan,
        "final_t2i_R10": final["t2i"]["R10"] if final else nan,
        "final_i2t_R1": final["i2t"]["R1"] if final else nan,
        "final_i2t_R5": final["i2t"]["R5"] if final else nan,
        "final_i2t_R10": final["i2t"]["R10"] if final else nan,
        "final_mean_R10": (
            0.5 * (final["t2i"]["R10"] + final["i2t"]["R10"])
            if final else nan
        ),
    }


def evaluate(
    args,
    train1,
    train2,
    test1,
    test2,
    idx1,
    idx2,
    quality,
    method,
    n_real,
    n_pseudo,
    arh_k,
    pool_size,
    coverage,
    mean_confidence,
    n_components,
    device,
):
    w1, w2 = fit_cross_cca(train1, train2, idx1, idx2, n_components)
    z = project(train1, train2, test1, test2, w1, w2)
    recall = compute_bidirectional_recall(z["test1"], z["test2"])
    print_recall(f"[{n_real} real + {n_pseudo} pseudo | {method}] SE -> CCA", recall)

    final = None
    if not args.skip_mmd:
        mmd = run_mmd(
            z["train1"],
            z["train2"],
            z["test1"],
            z["test2"],
            device=device,
            seed=args.seed,
            epochs=args.mmd_epochs,
            batch_size=args.mmd_batch_size,
            n_scales=args.mmd_scales,
        )
        final = compute_bidirectional_recall(mmd["test1"], mmd["test2"])
        print_recall(
            f"[{n_real} real + {n_pseudo} pseudo | {method}] SE -> CCA -> MMD",
            final,
        )

    return metric_row(
        args.data,
        args.seed,
        n_real,
        n_pseudo,
        method,
        arh_k,
        pool_size,
        coverage,
        mean_confidence,
        quality,
        recall,
        final,
    ), w1, w2


def self_check():
    anchors = np.eye(4, dtype=np.float32)
    x = np.array(
        [[1, 0.8, 0.1, 0], [0, 0.2, 1, 0.7], [0.9, 1, 0, 0.1]],
        dtype=np.float32,
    )
    for method in METHODS:
        p = anchor_profile(x, anchors, method, 2, 2.0, 0.1)
        i, t, _ = mutual_matches(p, p.copy())
        assert np.array_equal(i, t)
    print("Step15 self-check passed.")


def parse_args():
    p = argparse.ArgumentParser(description="SUE Step15 - ARH information ladder")
    p.add_argument("data", nargs="?", help="e.g. flickr30")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--pseudo_counts", type=int, nargs="+", default=[50])
    p.add_argument("--arh_k", type=int, default=10)
    p.add_argument("--rank_tau", type=float, default=2.0)
    p.add_argument("--soft_tau", type=float, default=0.1)
    p.add_argument("--with_controls", action="store_true")
    p.add_argument("--control_method", choices=METHODS, default="rank")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mmd_epochs", type=int, default=100)
    p.add_argument("--mmd_batch_size", type=int, default=32)
    p.add_argument("--mmd_scales", type=int, default=3)
    p.add_argument("--skip_mmd", action="store_true")
    p.add_argument("--cache", default=None)
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--self_check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        return args
    if not args.data:
        p.error("data is required unless --self_check is used")
    if args.arh_k <= 0 or args.rank_tau <= 0 or args.soft_tau <= 0:
        p.error("arh_k/rank_tau/soft_tau must be > 0")
    if any(n <= 0 for n in args.pseudo_counts):
        p.error("pseudo_counts must be > 0")
    return args


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    if args.self_check:
        self_check()
        return

    set_seed(args.seed)
    config = load_config(args.data)
    n_components = int(config["n_components"])
    cache = load_fixed_se_cache(args.data, args.seed, args.cache)
    train1, train2 = cache["train_se1"], cache["train_se2"]
    test1, test2 = cache["test_se1"], cache["test_se2"]
    pair_order = cache["pair_order"]

    for n_real in args.pairs:
        if n_real <= n_components or n_real > len(pair_order):
            raise ValueError(
                f"n_real must be in [{n_components + 1}, {len(pair_order)}]"
            )

    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    run_dir = Path(args.output_dir) / args.data / f"seed{args.seed}"
    diag_dir = run_dir / "diagnostics"
    run_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(exist_ok=True)

    encoded1 = torch.load(
        ROOT / "data" / args.data / "encoded1.pt", map_location="cpu"
    )
    n_original_train = len(encoded1) - int(config["n_test"])
    del encoded1
    ids1, ids2 = replay_weak_ids(n_original_train, len(pair_order), args.seed)
    if len(ids1) != len(train1) or len(ids2) != len(train2):
        raise RuntimeError("Weak-data replay != fixed-SE cache.")
    if not np.array_equal(ids1[pair_order], ids2[pair_order]):
        raise RuntimeError("Master-pair provenance replay failed.")

    candidate_idx = np.setdiff1d(
        np.arange(len(train1)), pair_order, assume_unique=True
    )
    rows = []
    rng = np.random.default_rng(args.seed + 15_000)

    print(f"Step15 | data={args.data} | seed={args.seed} | device={device}")
    print(f"pairs={args.pairs} | methods={args.methods} | pseudo={args.pseudo_counts}")
    print(f"arh_k={args.arh_k} | skip_mmd={args.skip_mmd}")
    print(f"unpaired candidates={len(candidate_idx)} | provenance=PASS")

    for n_real in args.pairs:
        print("\n" + "=" * 72)
        print(f"N_REAL = {n_real}")
        print("=" * 72)
        real_idx = pair_order[:n_real].copy()
        k = min(args.arh_k, n_real)

        row, w1, w2 = evaluate(
            args,
            train1,
            train2,
            test1,
            test2,
            real_idx,
            real_idx,
            empty_quality(),
            "real_only",
            n_real,
            0,
            k,
            0,
            0.0,
            float("nan"),
            n_components,
            device,
        )
        rows.append(row)
        np.savez_compressed(
            diag_dir / f"real{n_real}_real_only.npz",
            image_indices=real_idx,
            text_indices=real_idx,
            projection1=w1,
            projection2=w2,
        )

        for method in args.methods:
            image_profile = anchor_profile(
                train1[candidate_idx], train1[real_idx],
                method, k, args.rank_tau, args.soft_tau,
            )
            text_profile = anchor_profile(
                train2[candidate_idx], train2[real_idx],
                method, k, args.rank_tau, args.soft_tau,
            )
            image_local, text_local, confidence = mutual_matches(
                image_profile, text_profile
            )
            pseudo_image = candidate_idx[image_local]
            pseudo_text = candidate_idx[text_local]
            pool_size = len(pseudo_image)
            coverage = pool_size / len(candidate_idx)
            print(
                f"[{method}] mutual pool={pool_size} | coverage={coverage:.3f} | "
                f"best_conf={confidence[0] if pool_size else float('nan'):.6f}"
            )
            if not pool_size:
                continue

            for n_pseudo in args.pseudo_counts:
                if n_pseudo > pool_size:
                    print(f"[{method}] skip n_pseudo={n_pseudo}; pool={pool_size}")
                    continue
                pi = pseudo_image[:n_pseudo]
                pt = pseudo_text[:n_pseudo]
                q = pseudo_quality(
                    pi, pt, candidate_idx, ids1, ids2, train1, train2
                )
                mean_conf = float(confidence[:n_pseudo].mean())
                row, w1, w2 = evaluate(
                    args,
                    train1,
                    train2,
                    test1,
                    test2,
                    np.r_[real_idx, pi],
                    np.r_[real_idx, pt],
                    q,
                    method,
                    n_real,
                    n_pseudo,
                    k,
                    pool_size,
                    coverage,
                    mean_conf,
                    n_components,
                    device,
                )
                rows.append(row)
                np.savez_compressed(
                    diag_dir / f"real{n_real}_{method}_pseudo{n_pseudo}.npz",
                    real_indices=real_idx,
                    pseudo_image_indices=pi,
                    pseudo_text_indices=pt,
                    confidence=confidence[:n_pseudo],
                    projection1=w1,
                    projection2=w2,
                )

                if args.with_controls and method == args.control_method:
                    shuffled = rng.permutation(pt)
                    q = pseudo_quality(
                        pi, shuffled, candidate_idx, ids1, ids2, train1, train2
                    )
                    row, _, _ = evaluate(
                        args,
                        train1,
                        train2,
                        test1,
                        test2,
                        np.r_[real_idx, pi],
                        np.r_[real_idx, shuffled],
                        q,
                        f"{method}_shuffled",
                        n_real,
                        n_pseudo,
                        k,
                        pool_size,
                        coverage,
                        mean_conf,
                        n_components,
                        device,
                    )
                    rows.append(row)

                    oi, ot = oracle_pairs(
                        candidate_idx, ids1, ids2, n_pseudo, rng
                    )
                    q = pseudo_quality(
                        oi, ot, candidate_idx, ids1, ids2, train1, train2
                    )
                    row, _, _ = evaluate(
                        args,
                        train1,
                        train2,
                        test1,
                        test2,
                        np.r_[real_idx, oi],
                        np.r_[real_idx, ot],
                        q,
                        "oracle",
                        n_real,
                        n_pseudo,
                        k,
                        pool_size,
                        coverage,
                        float("nan"),
                        n_components,
                        device,
                    )
                    rows.append(row)

        save_results_csv(rows, run_dir / "results.csv")

    print(f"\nStep15 completed. Results: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
