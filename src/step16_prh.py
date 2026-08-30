"""SUE Step16 - PRH replacement and boundary study.

Fixed SE -> PRH pseudo-pairs -> CCA -> optional unchanged SUE MMD.
PRH methods: GW, pairwise-distance profile matching, partial GW.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import ot
import torch
from sklearn.cross_decomposition import CCA

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
DEFAULT_OUT = ROOT / "results" / "step16"
DEFAULT_PAIRS = [500, 300, 200, 175, 150, 125, 100, 75, 50, 25, 10]
METHODS = ("gw", "distance", "partial_gw")


def pairwise_cost(x):
    """Cosine-distance structure, normalized to [0, 1]."""
    z = l2_normalize(x)
    cost = np.maximum(1.0 - z @ z.T, 0.0).astype(np.float64)
    scale = float(cost.max())
    return cost / scale if scale > 0 else cost


def distance_profile(x, anchors):
    """Global pairwise-distance relations to the paired anchor set."""
    d = 1.0 - l2_normalize(x) @ l2_normalize(anchors).T
    d -= d.mean(axis=1, keepdims=True)
    d /= d.std(axis=1, keepdims=True) + 1e-8
    return l2_normalize(d.astype(np.float32))


def mutual_similarity_matches(image_profile, text_profile):
    """Mutual nearest matches; cosine top1-top2 margin is confidence."""
    sim = l2_normalize(image_profile) @ l2_normalize(text_profile).T
    best_text = sim.argmax(axis=1)
    best_image = sim.argmax(axis=0)
    rows = np.arange(len(image_profile))
    keep = best_image[best_text] == rows

    top1 = sim[rows, best_text]
    if sim.shape[1] > 1:
        top2 = np.partition(sim, -2, axis=1)[:, -2]
    else:
        top2 = np.zeros_like(top1)
    confidence = (top1 - top2)[keep]

    image_local = rows[keep]
    text_local = best_text[keep]
    order = np.argsort(-confidence)
    return image_local[order], text_local[order], confidence[order]


def solve_gw(image, text, max_iter):
    """Classic GW on intra-modal pairwise-distance matrices."""
    c1, c2 = pairwise_cost(image), pairwise_cost(text)
    p = np.full(len(image), 1.0 / len(image), dtype=np.float64)
    q = np.full(len(text), 1.0 / len(text), dtype=np.float64)
    return np.asarray(
        ot.gromov.gromov_wasserstein(
            c1,
            c2,
            p=p,
            q=q,
            loss_fun="square_loss",
            max_iter=max_iter,
            tol_rel=1e-7,
            tol_abs=1e-7,
            log=False,
        )
    )

# Old version makes some mistakes
# def solve_partial_gw(image, text, mass, max_iter):
#     """Partial GW; fallback keeps compatibility with POT 0.9.x."""
#     c1, c2 = pairwise_cost(image), pairwise_cost(text)
#     p = np.full(len(image), 1.0 / len(image), dtype=np.float64)
#     q = np.full(len(text), 1.0 / len(text), dtype=np.float64)
#     solver = getattr(ot.gromov, "partial_gromov_wasserstein", None)
#     if solver is None:
#         solver = ot.partial.partial_gromov_wasserstein
#     return np.asarray(
#         solver(
#             c1,
#             c2,
#             p=p,
#             q=q,
#             m=mass,
#             numItermax=max_iter,
#             tol=1e-7,
#             log=False,
#         )
#     )

def solve_partial_gw(image, text, mass, max_iter):
    """Partial GW; mass=1 uses full GW to avoid POT feasibility roundoff."""
    mass = float(mass)

    if not 0.0 < mass <= 1.0:
        raise ValueError(f"partial_mass must be in (0, 1], got {mass}")

    if np.isclose(mass, 1.0):
        return solve_gw(image, text, max_iter)

    c1, c2 = pairwise_cost(image), pairwise_cost(text)

    p = np.full(len(image), 1.0 / len(image), dtype=np.float64)
    q = np.full(len(text), 1.0 / len(text), dtype=np.float64)

    p /= p.sum()
    q /= q.sum()

    feasible_mass = min(
        mass,
        np.nextafter(min(p.sum(), q.sum()), 0.0),
    )

    solver = getattr(
        ot.gromov,
        "partial_gromov_wasserstein",
        None,
    )

    if solver is None:
        solver = ot.partial.partial_gromov_wasserstein

    return np.asarray(
        solver(
            c1,
            c2,
            p=p,
            q=q,
            m=feasible_mass,
            numItermax=max_iter,
            tol=1e-7,
            log=False,
        )
    )

def transport_matches(plan):
    """Turn a GW transport plan into confidence-ranked mutual pseudo-pairs."""
    best_text = plan.argmax(axis=1)
    best_image = plan.argmax(axis=0)
    rows = np.arange(plan.shape[0])
    row_mass = plan.sum(axis=1)
    top1 = plan[rows, best_text]
    if plan.shape[1] > 1:
        top2 = np.partition(plan, -2, axis=1)[:, -2]
    else:
        top2 = np.zeros_like(top1)

    keep = (
        (best_image[best_text] == rows)
        & (row_mass > 1e-12)
        & (top1 > 0)
    )
    confidence = ((top1 - top2) / np.maximum(row_mass, 1e-12))[keep]
    image_local = rows[keep]
    text_local = best_text[keep]
    order = np.argsort(-confidence)
    return image_local[order], text_local[order], confidence[order]


def fit_cross_cca(train1, train2, idx1, idx2, n_components):
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
    args,
    method,
    n_real,
    n_pseudo,
    partial_mass,
    transport_mass,
    pool_size,
    coverage,
    mean_confidence,
    quality,
    recall,
    final,
):
    nan = float("nan")
    return {
        "dataset": args.data,
        "seed": args.seed,
        "n_real": n_real,
        "n_pseudo": n_pseudo,
        "method": method,
        "gw_pool": args.gw_pool,
        "partial_mass": partial_mass,
        "transport_mass": transport_mass,
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
            if final
            else nan
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
    partial_mass,
    transport_mass,
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

    row = metric_row(
        args,
        method,
        n_real,
        n_pseudo,
        partial_mass,
        transport_mass,
        pool_size,
        coverage,
        mean_confidence,
        quality,
        recall,
        final,
    )
    return row, w1, w2


def self_check():
    x = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    c = pairwise_cost(x)
    assert c.shape == (3, 3) and np.allclose(np.diag(c), 0)

    plan = np.eye(3, dtype=np.float64) / 3
    i, t, conf = transport_matches(plan)
    assert np.array_equal(i, t) and len(conf) == 3

    p = distance_profile(x, x)
    i, t, _ = mutual_similarity_matches(p, p.copy())
    assert np.array_equal(i, t)
    print("Step16 self-check passed.")


def parse_args():
    p = argparse.ArgumentParser(description="SUE Step16 - PRH replacement/boundary")
    p.add_argument("data", nargs="?", help="e.g. flickr30")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--pseudo_counts", type=int, nargs="+", default=[50])
    p.add_argument("--gw_pool", type=int, default=400)
    p.add_argument("--gw_max_iter", type=int, default=200)
    p.add_argument("--partial_mass", type=float, default=0.5)
    p.add_argument("--with_controls", action="store_true")
    p.add_argument("--control_method", choices=METHODS, default="gw")
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
    if args.gw_pool <= 1 or args.gw_max_iter <= 0:
        p.error("gw_pool must be > 1 and gw_max_iter must be > 0")
    if not 0 < args.partial_mass <= 1:
        p.error("partial_mass must be in (0, 1]")
    if any(n <= 0 for n in args.pseudo_counts):
        p.error("pseudo_counts must be > 0")
    return args


def save_diag(path, real_idx, pi, pt, confidence, w1, w2, plan=None):
    payload = {
        "real_indices": real_idx,
        "pseudo_image_indices": pi,
        "pseudo_text_indices": pt,
        "confidence": confidence,
        "projection1": w1,
        "projection2": w2,
    }
    if plan is not None:
        payload["transport_plan"] = plan.astype(np.float32)
    np.savez_compressed(path, **payload)


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

    encoded1 = torch.load(ROOT / "data" / args.data / "encoded1.pt", map_location="cpu")
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
    rng = np.random.default_rng(args.seed + 16_000)
    pool_n = min(args.gw_pool, len(candidate_idx))
    structural_idx = np.sort(rng.choice(candidate_idx, size=pool_n, replace=False))

    # Ponytail: GW is independent of n_real, so solve it once per seed.
    precomputed = {}
    if "gw" in args.methods:
        print(f"Solving GW once on fixed structural pool n={pool_n}...")
        plan = solve_gw(train1[structural_idx], train2[structural_idx], args.gw_max_iter)
        il, tl, conf = transport_matches(plan)
        precomputed["gw"] = (structural_idx[il], structural_idx[tl], conf, plan)

    if "partial_gw" in args.methods:
        print(
            f"Solving Partial GW once on fixed structural pool n={pool_n}, "
            f"mass={args.partial_mass}..."
        )
        plan = solve_partial_gw(
            train1[structural_idx],
            train2[structural_idx],
            args.partial_mass,
            args.gw_max_iter,
        )
        il, tl, conf = transport_matches(plan)
        precomputed["partial_gw"] = (
            structural_idx[il], structural_idx[tl], conf, plan
        )

    print(f"Step16 | data={args.data} | seed={args.seed} | device={device}")
    print(f"pairs={args.pairs} | methods={args.methods} | pseudo={args.pseudo_counts}")
    print(
        f"gw_pool={pool_n} | partial_mass={args.partial_mass} | "
        f"skip_mmd={args.skip_mmd}"
    )
    print(f"unpaired candidates={len(candidate_idx)} | provenance=PASS")

    rows = []
    control_rng = np.random.default_rng(args.seed + 160_000)

    for n_real in args.pairs:
        print("\n" + "=" * 72)
        print(f"N_REAL = {n_real}")
        print("=" * 72)
        real_idx = pair_order[:n_real].copy()

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
            float("nan"),
            float("nan"),
            0,
            0.0,
            float("nan"),
            n_components,
            device,
        )
        rows.append(row)
        save_diag(
            diag_dir / f"real{n_real}_real_only.npz",
            real_idx,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            w1,
            w2,
        )

        for method in args.methods:
            if method == "distance":
                image_profile = distance_profile(
                    train1[structural_idx], train1[real_idx]
                )
                text_profile = distance_profile(
                    train2[structural_idx], train2[real_idx]
                )
                il, tl, confidence = mutual_similarity_matches(
                    image_profile, text_profile
                )
                pseudo_image = structural_idx[il]
                pseudo_text = structural_idx[tl]
                plan = None
                partial_mass = float("nan")
                transport_mass = float("nan")
            else:
                pseudo_image, pseudo_text, confidence, plan = precomputed[method]
                partial_mass = args.partial_mass if method == "partial_gw" else float("nan")
                transport_mass = float(plan.sum())

            pool_size = len(pseudo_image)
            coverage = pool_size / pool_n
            print(
                f"[{method}] pseudo pool={pool_size} | coverage={coverage:.3f} | "
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
                conf = confidence[:n_pseudo]
                quality = pseudo_quality(
                    pi, pt, candidate_idx, ids1, ids2, train1, train2
                )
                row, w1, w2 = evaluate(
                    args,
                    train1,
                    train2,
                    test1,
                    test2,
                    np.r_[real_idx, pi],
                    np.r_[real_idx, pt],
                    quality,
                    method,
                    n_real,
                    n_pseudo,
                    partial_mass,
                    transport_mass,
                    pool_size,
                    coverage,
                    float(conf.mean()),
                    n_components,
                    device,
                )
                rows.append(row)
                save_diag(
                    diag_dir / f"real{n_real}_{method}_pseudo{n_pseudo}.npz",
                    real_idx,
                    pi,
                    pt,
                    conf,
                    w1,
                    w2,
                    plan,
                )

                if args.with_controls and method == args.control_method:
                    shuffled = control_rng.permutation(pt)
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
                        partial_mass,
                        transport_mass,
                        pool_size,
                        coverage,
                        float(conf.mean()),
                        n_components,
                        device,
                    )
                    rows.append(row)

                    oi, ot = oracle_pairs(
                        candidate_idx, ids1, ids2, n_pseudo, control_rng
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
                        float("nan"),
                        float("nan"),
                        len(candidate_idx),
                        n_pseudo / len(candidate_idx),
                        float("nan"),
                        n_components,
                        device,
                    )
                    rows.append(row)

        save_results_csv(rows, run_dir / "results.csv")

    print(f"\nStep16 completed. Results: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
