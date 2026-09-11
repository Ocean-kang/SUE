"""SUE Step20 - Relational Pair Information Generalization.

Fixed SE -> CCA -> optional unchanged SUE MMD.
Conditions: real-only / relation pseudo-pairs / shuffled / oracle.

Main:
  pairs = 10 25 50 75 100 150 200 300 500
  pseudo = 75

Saturation:
  choose representative real-pair budgets after Main,
  pseudo = 25 50 75 100 150
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pair_removal import (
    compute_bidirectional_recall,
    load_config,
    load_fixed_se_cache,
    print_recall,
    run_mmd,
    set_seed,
)
from step17_functional_map import fit_cross_cca, project
from step8_correspondence_replacement_v2 import (
    mutual_matches,
    oracle_pairs,
    pseudo_quality,
    rank_signature,
    replay_weak_ids,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "step20"
DEFAULT_PAIRS = [10, 25, 50, 75, 100, 150, 200, 300, 500]


def empty_quality():
    nan = float("nan")
    return {
        "exact_precision": nan,
        "gt_available_rate": nan,
        "semantic_hit_R10": nan,
        "semantic_hit_R100": nan,
        "median_gt_rank": nan,
    }


def stable_rng(seed, n_real, n_pseudo, stream):
    mixed = (
        int(seed) * 1_000_003
        + int(n_real) * 10_007
        + int(n_pseudo) * 1_009
        + int(stream) * 97
        + 17
    ) % (2**32)
    return np.random.default_rng(mixed)


def recall_fields(prefix, recall):
    names = ("R1", "R5", "R10")
    if recall is None:
        return {
            **{f"{prefix}_{d}_{r}": np.nan for d in ("t2i", "i2t") for r in names},
            **{f"{prefix}_mean_{r}": np.nan for r in names},
            f"{prefix}_mR": np.nan,
        }

    row = {}
    for r in names:
        row[f"{prefix}_t2i_{r}"] = recall["t2i"][r]
        row[f"{prefix}_i2t_{r}"] = recall["i2t"][r]
        row[f"{prefix}_mean_{r}"] = 0.5 * (
            recall["t2i"][r] + recall["i2t"][r]
        )

    row[f"{prefix}_mR"] = np.mean([
        recall["t2i"]["R1"], recall["t2i"]["R5"], recall["t2i"]["R10"],
        recall["i2t"]["R1"], recall["i2t"]["R5"], recall["i2t"]["R10"],
    ])
    return row


def evaluate(
    args, train1, train2, test1, test2,
    idx1, idx2, n_components, device, title,
):
    w1, w2 = fit_cross_cca(
        train1, train2, idx1, idx2, n_components
    )
    z = project(
        train1, train2, test1, test2, w1, w2
    )

    cca = compute_bidirectional_recall(
        z["test1"], z["test2"]
    )
    print_recall(title + " | SE -> CCA", cca)

    final = None
    if not args.skip_mmd:
        m = run_mmd(
            z["train1"], z["train2"],
            z["test1"], z["test2"],
            device=device,
            seed=args.seed,
            epochs=args.mmd_epochs,
            batch_size=args.mmd_batch_size,
            n_scales=args.mmd_scales,
        )
        final = compute_bidirectional_recall(
            m["test1"], m["test2"]
        )
        print_recall(
            title + " | SE -> CCA -> MMD", final
        )

    return cca, final, w1, w2


def result_row(
    args, condition, n_real, n_pseudo,
    pool_size, mean_confidence, quality,
    cca=None, final=None, status="ok",
):
    return {
        "dataset": args.data,
        "mode": args.mode,
        "seed": args.seed,
        "condition": condition,
        "status": status,
        "n_real": n_real,
        "n_pseudo": n_pseudo,
        "pseudo_pool_size": pool_size,
        "mean_confidence": mean_confidence,
        **quality,
        **recall_fields("cca", cca),
        **recall_fields("final", final),
    }


def unavailable_row(args, condition, n_real, n_pseudo, pool_size):
    return result_row(
        args, condition, n_real, n_pseudo,
        pool_size, np.nan, empty_quality(),
        status="insufficient_pseudo_pool",
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="SUE Step20 - relational pair generalization"
    )
    p.add_argument("data", help="flickr30 or mscoco")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mode",
        choices=("main", "saturation"),
        default="main",
    )
    p.add_argument(
        "--pairs",
        type=int,
        nargs="+",
        default=DEFAULT_PAIRS,
    )
    p.add_argument(
        "--pseudo_counts",
        type=int,
        nargs="+",
        default=[75],
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mmd_epochs", type=int, default=100)
    p.add_argument("--mmd_batch_size", type=int, default=32)
    p.add_argument("--mmd_scales", type=int, default=3)
    p.add_argument("--skip_mmd", action="store_true")
    p.add_argument("--cache", default=None)
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    return p.parse_args()


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.data)
    n_components = int(config["n_components"])

    cache = load_fixed_se_cache(
        args.data, args.seed, args.cache
    )
    train1, train2 = cache["train_se1"], cache["train_se2"]
    test1, test2 = cache["test_se1"], cache["test_se2"]
    pair_order = cache["pair_order"]

    pairs = list(dict.fromkeys(args.pairs))
    pseudos = list(dict.fromkeys(args.pseudo_counts))

    for n in pairs:
        if n <= n_components or n > len(pair_order):
            raise ValueError(
                f"n_real={n}; expected "
                f"{n_components + 1} <= n_real <= {len(pair_order)}"
            )
    if any(n <= 0 for n in pseudos):
        raise ValueError("pseudo_counts must be positive.")

    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )

    run_dir = (
        Path(args.output_dir)
        / args.mode
        / args.data
        / f"seed{args.seed}"
    )
    diag_dir = run_dir / "diagnostics"
    run_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(exist_ok=True)

    # Same leakage-free protocol as Step8/9:
    # the complete 500-pair master block is never a pseudo candidate.
    candidate_idx = np.setdiff1d(
        np.arange(len(train1)),
        pair_order,
        assume_unique=True,
    )

    encoded1 = torch.load(
        ROOT / "data" / args.data / "encoded1.pt",
        map_location="cpu",
    )
    n_original_train = len(encoded1) - int(config["n_test"])
    del encoded1

    ids1, ids2 = replay_weak_ids(
        n_original_train,
        len(pair_order),
        args.seed,
    )
    if len(ids1) != len(train1) or len(ids2) != len(train2):
        raise RuntimeError("Weak-data replay != fixed-SE cache.")
    if not np.array_equal(ids1[pair_order], ids2[pair_order]):
        raise RuntimeError("Master-pair provenance replay failed.")

    print(
        f"Step20 | data={args.data} | seed={args.seed} | "
        f"mode={args.mode} | device={device}"
    )
    print(f"pairs={pairs} | pseudo_counts={pseudos}")
    print(
        f"master={len(pair_order)} | candidates={len(candidate_idx)} | "
        f"provenance=PASS | skip_mmd={args.skip_mmd}"
    )

    rows = []

    for n_real in pairs:
        print("\n" + "=" * 72)
        print(f"N_REAL = {n_real}")
        print("=" * 72)

        real = pair_order[:n_real].copy()

        # Relation signatures are rebuilt at every real-pair budget.
        image_sig = rank_signature(
            train1[candidate_idx], train1[real]
        )
        text_sig = rank_signature(
            train2[candidate_idx], train2[real]
        )
        il, tl, confidence = mutual_matches(
            image_sig, text_sig
        )
        pseudo_image = candidate_idx[il]
        pseudo_text = candidate_idx[tl]
        pool_size = len(pseudo_image)

        print(f"Mutual relation pool: {pool_size}")

        # Real-only baseline: once per N_real.
        cca, final, w1, w2 = evaluate(
            args, train1, train2, test1, test2,
            real, real, n_components, device,
            f"[real_only | {n_real} real]",
        )
        rows.append(result_row(
            args, "real_only", n_real, 0,
            pool_size, np.nan, empty_quality(),
            cca, final,
        ))
        np.savez_compressed(
            diag_dir / f"real{n_real}_real_only.npz",
            real_indices=real,
            projection1=w1,
            projection2=w2,
        )

        for n_pseudo in pseudos:
            print(
                f"\n--- N_REAL={n_real}, N_PSEUDO={n_pseudo} ---"
            )

            # Relation + shuffled need the requested fixed pseudo budget.
            if pool_size < n_pseudo:
                print(
                    f"Relation failure: requested={n_pseudo}, "
                    f"available={pool_size}"
                )
                rows.append(unavailable_row(
                    args, "relation", n_real, n_pseudo, pool_size
                ))
                rows.append(unavailable_row(
                    args, "shuffled", n_real, n_pseudo, pool_size
                ))
            else:
                pi = pseudo_image[:n_pseudo]
                pt = pseudo_text[:n_pseudo]
                mean_conf = float(
                    confidence[:n_pseudo].mean()
                )

                q = pseudo_quality(
                    pi, pt, candidate_idx,
                    ids1, ids2, train1, train2,
                )
                cca, final, w1, w2 = evaluate(
                    args, train1, train2, test1, test2,
                    np.r_[real, pi], np.r_[real, pt],
                    n_components, device,
                    f"[relation | {n_real}+{n_pseudo}]",
                )
                rows.append(result_row(
                    args, "relation", n_real, n_pseudo,
                    pool_size, mean_conf, q, cca, final,
                ))

                rng = stable_rng(
                    args.seed, n_real, n_pseudo, stream=1
                )
                shuffled_pt = rng.permutation(pt)
                q_shuf = pseudo_quality(
                    pi, shuffled_pt, candidate_idx,
                    ids1, ids2, train1, train2,
                )
                cca_s, final_s, _, _ = evaluate(
                    args, train1, train2, test1, test2,
                    np.r_[real, pi],
                    np.r_[real, shuffled_pt],
                    n_components, device,
                    f"[shuffled | {n_real}+{n_pseudo}]",
                )
                rows.append(result_row(
                    args, "shuffled", n_real, n_pseudo,
                    pool_size, mean_conf, q_shuf,
                    cca_s, final_s,
                ))

                np.savez_compressed(
                    diag_dir
                    / f"real{n_real}_pseudo{n_pseudo}_relation.npz",
                    real_indices=real,
                    pseudo_image_indices=pi,
                    pseudo_text_indices=pt,
                    confidence=confidence[:n_pseudo],
                    projection1=w1,
                    projection2=w2,
                )

            # Oracle does not depend on the relation pool.
            oracle_rng = stable_rng(
                args.seed, n_real, n_pseudo, stream=2
            )
            oi, ot = oracle_pairs(
                candidate_idx, ids1, ids2,
                n_pseudo, oracle_rng,
            )
            q_oracle = pseudo_quality(
                oi, ot, candidate_idx,
                ids1, ids2, train1, train2,
            )
            cca_o, final_o, _, _ = evaluate(
                args, train1, train2, test1, test2,
                np.r_[real, oi], np.r_[real, ot],
                n_components, device,
                f"[oracle | {n_real}+{n_pseudo}]",
            )
            rows.append(result_row(
                args, "oracle", n_real, n_pseudo,
                pool_size, np.nan, q_oracle,
                cca_o, final_o,
            ))

        pd.DataFrame(rows).to_csv(
            run_dir / "results.csv", index=False
        )

    print(f"\nStep20 completed: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
