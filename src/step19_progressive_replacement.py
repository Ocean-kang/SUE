"""SUE Step19 - Progressive functional replacement.

Fixed SE -> {CCA, direct ridge map} x {real-only, ARH TopK constraints}
         -> optional unchanged SUE MMD.

Purpose:
1) separate structural substitution from estimator efficiency;
2) test whether the two effects are complementary;
3) measure pair-performance left shift before and after MMD.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

from pair_removal import (
    compute_bidirectional_recall,
    load_config,
    load_fixed_se_cache,
    print_recall,
    run_mmd,
    save_results_csv,
    set_seed,
)
from step15_arh import anchor_profile, mutual_matches
from step17_functional_map import fit_cross_cca, project, recall_fields
from step8_correspondence_replacement_v2 import pseudo_quality, replay_weak_ids

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "step19"
DEFAULT_PAIRS = [500, 300, 200, 175, 150, 125, 100, 75, 50, 25, 10]


def target_basis(x, n_components):
    """Pair-free target basis; keeps direct-map output dimension matched to CCA."""
    gram = x.astype(np.float64).T @ x.astype(np.float64)
    _, vectors = np.linalg.eigh(gram)
    return vectors[:, -n_components:][:, ::-1].astype(np.float32)


def fit_direct_map(x1, x2, idx1, idx2, target_w, ridge):
    """One-sided ridge/FM-style map into a fixed target basis."""
    a = x1[idx1].astype(np.float64)
    b = (x2[idx2] @ target_w).astype(np.float64)
    eye = np.eye(a.shape[1], dtype=np.float64)
    w = np.linalg.solve(a.T @ a + ridge * eye, a.T @ b)
    fit = np.linalg.norm(a @ w - b) / (np.linalg.norm(b) + 1e-12)
    cond = np.linalg.cond(a.T @ a + ridge * eye)
    return w.astype(np.float32), float(fit), float(cond)


def direct_project(train1, train2, test1, test2, w, target_w):
    """Project both modalities to the same dimension-matched target space."""
    return project(train1, train2, test1, test2, w, target_w)


def evaluate(z, args, device, title):
    recall = compute_bidirectional_recall(z["test1"], z["test2"])
    print_recall(title, recall)
    final = None
    if not args.skip_mmd:
        m = run_mmd(
            z["train1"], z["train2"], z["test1"], z["test2"],
            device=device,
            seed=args.seed,
            epochs=args.mmd_epochs,
            batch_size=args.mmd_batch_size,
            n_scales=args.mmd_scales,
        )
        final = compute_bidirectional_recall(m["test1"], m["test2"])
        print_recall(title + " -> MMD", final)
    return recall, final


def make_row(args, n_real, n_arh, stage, arh_k, pool_size, quality,
             map_fit, map_cond, recall, final):
    nan = float("nan")
    return {
        "dataset": args.data,
        "seed": args.seed,
        "n_real": n_real,
        "n_arh": n_arh,
        "stage": stage,
        "estimator": "fm_direct" if "fm" in stage else "cca",
        "structure": "arh_topk" if "arh" in stage else "none",
        "arh_k": arh_k,
        "arh_target": args.arh_pseudo,
        "arh_pool_size": pool_size,
        "map_ridge": args.map_ridge,
        "map_fit_error": map_fit,
        "map_condition": map_cond,
        **quality,
        **recall_fields("align", recall),
        **recall_fields("final", final),
        "gain_vs_cca": nan,
        "factorial_synergy": nan,
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


def mean_r10(recall):
    return 0.5 * (recall["t2i"]["R10"] + recall["i2t"]["R10"])


def run_pair_budget(args, train1, train2, test1, test2, pair_order,
                    candidate_idx, ids1, ids2, n_components, target_w,
                    device, diag_dir):
    n_real = args.n_real
    real = pair_order[:n_real].copy()

    # Avoid Step15's K=N degeneration at 10 pairs; unchanged for N>=20.
    arh_k = min(args.arh_k, max(1, n_real // 2))
    ip = anchor_profile(
        train1[candidate_idx], train1[real],
        "topk", arh_k, args.rank_tau, args.soft_tau,
    )
    tp = anchor_profile(
        train2[candidate_idx], train2[real],
        "topk", arh_k, args.rank_tau, args.soft_tau,
    )
    il, tl, conf = mutual_matches(ip, tp)
    pool_size = len(il)
    n_arh = min(args.arh_pseudo, pool_size)

    if n_arh == 0:
        raise RuntimeError(f"No ARH matches for n_real={n_real}")

    pi = candidate_idx[il[:n_arh]]
    pt = candidate_idx[tl[:n_arh]]
    quality = pseudo_quality(pi, pt, candidate_idx, ids1, ids2, train1, train2)

    mixed_i = np.r_[real, pi]
    mixed_t = np.r_[real, pt]
    print(
        f"\nN={n_real} | ARH TopK={arh_k} | "
        f"pool={pool_size} | use={n_arh}"
    )

    rows = []
    projections = {}

    # P0: ordinary paired CCA.
    w1, w2 = fit_cross_cca(train1, train2, real, real, n_components)
    z = project(train1, train2, test1, test2, w1, w2)
    r0, f0 = evaluate(z, args, device, f"[P0 | {n_real} real] SE -> CCA")
    rows.append(make_row(
        args, n_real, 0, "p0_cca", arh_k, pool_size, empty_quality(),
        float("nan"), float("nan"), r0, f0,
    ))
    projections["p0_cca"] = (w1, w2)

    # P1: structural substitute while keeping the original CCA estimator.
    aw1, aw2 = fit_cross_cca(train1, train2, mixed_i, mixed_t, n_components)
    az = project(train1, train2, test1, test2, aw1, aw2)
    r1, f1 = evaluate(
        az, args, device,
        f"[P1 | {n_real} real + {n_arh} ARH] SE -> CCA",
    )
    rows.append(make_row(
        args, n_real, n_arh, "p1_arh_cca", arh_k, pool_size, quality,
        float("nan"), float("nan"), r1, f1,
    ))
    projections["p1_arh_cca"] = (aw1, aw2)

    # P2: same real pairs, more pair-efficient directed estimator.
    fw, fit2, cond2 = fit_direct_map(
        train1, train2, real, real, target_w, args.map_ridge
    )
    fz = direct_project(train1, train2, test1, test2, fw, target_w)
    r2, f2 = evaluate(fz, args, device, f"[P2 | {n_real} real] SE -> FM-direct")
    rows.append(make_row(
        args, n_real, 0, "p2_fm", arh_k, pool_size, empty_quality(),
        fit2, cond2, r2, f2,
    ))
    projections["p2_fm"] = (fw, target_w)

    # P3: structural constraints + pair-efficient estimator.
    afw, fit3, cond3 = fit_direct_map(
        train1, train2, mixed_i, mixed_t, target_w, args.map_ridge
    )
    afz = direct_project(train1, train2, test1, test2, afw, target_w)
    r3, f3 = evaluate(
        afz, args, device,
        f"[P3 | {n_real} real + {n_arh} ARH] SE -> FM-direct",
    )
    rows.append(make_row(
        args, n_real, n_arh, "p3_arh_fm", arh_k, pool_size, quality,
        fit3, cond3, r3, f3,
    ))
    projections["p3_arh_fm"] = (afw, target_w)

    base = mean_r10(r0)
    values = [mean_r10(r) for r in (r0, r1, r2, r3)]
    synergy = values[3] - values[1] - values[2] + values[0]
    for row, value in zip(rows, values):
        row["gain_vs_cca"] = value - base
    rows[-1]["factorial_synergy"] = synergy

    np.savez_compressed(
        diag_dir / f"pairs{n_real}.npz",
        real_indices=real,
        arh_image_indices=pi,
        arh_text_indices=pt,
        arh_confidence=conf[:n_arh],
        p0_w1=projections["p0_cca"][0],
        p0_w2=projections["p0_cca"][1],
        p1_w1=projections["p1_arh_cca"][0],
        p1_w2=projections["p1_arh_cca"][1],
        p2_w1=projections["p2_fm"][0],
        p3_w1=projections["p3_arh_fm"][0],
    )
    return rows


def parse_args():
    p = argparse.ArgumentParser(
        description="SUE Step19 - progressive functional replacement"
    )
    p.add_argument("data", help="e.g. flickr30")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--arh_pseudo", type=int, default=75)
    p.add_argument("--arh_k", type=int, default=10)
    p.add_argument("--rank_tau", type=float, default=2.0)
    p.add_argument("--soft_tau", type=float, default=0.1)
    p.add_argument("--map_ridge", type=float, default=1e-3)
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
    cache = load_fixed_se_cache(args.data, args.seed, args.cache)
    train1, train2 = cache["train_se1"], cache["train_se2"]
    test1, test2 = cache["test_se1"], cache["test_se2"]
    pair_order = cache["pair_order"]

    if any(n <= n_components or n > len(pair_order) for n in args.pairs):
        raise ValueError(
            f"pairs must be in [{n_components + 1}, {len(pair_order)}]"
        )

    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    mode = "align_only" if args.skip_mmd else "full_mmd"
    run_dir = Path(args.output_dir) / mode / args.data / f"seed{args.seed}"
    diag_dir = run_dir / "diagnostics"
    run_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(exist_ok=True)

    # Same provenance replay used by Steps 15/17.
    encoded1 = torch.load(
        ROOT / "data" / args.data / "encoded1.pt", map_location="cpu"
    )
    n_original_train = len(encoded1) - int(config["n_test"])
    del encoded1
    ids1, ids2 = replay_weak_ids(
        n_original_train, len(pair_order), args.seed
    )
    if not np.array_equal(ids1[pair_order], ids2[pair_order]):
        raise RuntimeError("Master-pair provenance replay failed.")

    # Withheld master pairs are never pseudo candidates.
    candidate_idx = np.setdiff1d(
        np.arange(len(train1)), pair_order, assume_unique=True
    )

    print(
        f"Step19 | data={args.data} | seed={args.seed} | device={device}\n"
        f"pairs={args.pairs} | ARH=TopK/{args.arh_pseudo} | "
        f"ridge={args.map_ridge} | skip_mmd={args.skip_mmd}"
    )

    target_w = target_basis(train2, n_components)

    rows = []
    for n_real in args.pairs:
        args.n_real = n_real
        rows.extend(run_pair_budget(
            args, train1, train2, test1, test2, pair_order,
            candidate_idx, ids1, ids2, n_components, target_w,
            device, diag_dir,
        ))
        save_results_csv(rows, run_dir / "results.csv")

    print(f"\nStep19 completed: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
