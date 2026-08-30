"""SUE Step17 - Functional Map pair-information propagation.

Fixed SE -> anchor FM -> pseudo pairs -> CCA -> optional unchanged SUE MMD.
Methods: fm, fm_mutual, fm_confidence + optional shuffled/oracle controls.
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
DEFAULT_OUT = ROOT / "results" / "step17"
DEFAULT_PAIRS = [500, 300, 200, 175, 150, 125, 100, 75, 50, 25, 10]
METHODS = ("fm", "fm_mutual", "fm_confidence")


def fit_fm(x1, x2, anchors, ridge):
    """Fit bidirectional d x d functional maps from real anchor pairs."""
    a = x1[anchors].astype(np.float64)
    b = x2[anchors].astype(np.float64)
    eye = np.eye(a.shape[1], dtype=np.float64)
    c12 = np.linalg.solve(a.T @ a + ridge * eye, a.T @ b)
    c21 = np.linalg.solve(b.T @ b + ridge * eye, b.T @ a)

    eps = 1e-12
    fit12 = np.linalg.norm(a @ c12 - b) / (np.linalg.norm(b) + eps)
    fit21 = np.linalg.norm(b @ c21 - a) / (np.linalg.norm(a) + eps)
    cyc1 = np.linalg.norm(a @ c12 @ c21 - a) / (np.linalg.norm(a) + eps)
    cyc2 = np.linalg.norm(b @ c21 @ c12 - b) / (np.linalg.norm(b) + eps)
    ortho = np.linalg.norm(c12.T @ c12 - eye) / np.sqrt(a.shape[1])

    diag = {
        "fm_fit_error": 0.5 * (fit12 + fit21),
        "fm_cycle_error": 0.5 * (cyc1 + cyc2),
        "fm_ortho_error": ortho,
    }
    return c12.astype(np.float32), c21.astype(np.float32), diag


def _nn2(query, gallery):
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(l2_normalize(gallery))
    return nn.kneighbors(l2_normalize(query))


def _greedy_unique(src, tgt, score):
    """Keep high-score one-to-one pairs without a heavy assignment solver."""
    order = np.argsort(-score)
    used_tgt = set()
    keep = []
    for k in order:
        j = int(tgt[k])
        if j in used_tgt:
            continue
        used_tgt.add(j)
        keep.append(k)
    keep = np.asarray(keep, dtype=np.int64)
    return src[keep], tgt[keep], score[keep]


def fm_matches(x1, x2, candidate_idx, c12, c21, method):
    """Generate pseudo pairs from FM-mapped candidate samples."""
    a = x1[candidate_idx]
    b = x2[candidate_idx]
    mapped_a = a @ c12
    mapped_b = b @ c21

    d12, best12 = _nn2(mapped_a, b)
    _, best21 = _nn2(mapped_b, a)

    src = np.arange(len(candidate_idx), dtype=np.int64)
    tgt = best12[:, 0].astype(np.int64)
    margin = d12[:, 1] - d12[:, 0]
    similarity = 1.0 - d12[:, 0]

    cycle = np.linalg.norm(a @ c12 @ c21 - a, axis=1)
    cycle /= np.linalg.norm(a, axis=1) + 1e-12

    if method == "fm":
        score = similarity
        src, tgt, score = _greedy_unique(src, tgt, score)
    elif method == "fm_confidence":
        score = margin / (1.0 + cycle)
        src, tgt, score = _greedy_unique(src, tgt, score)
    elif method == "fm_mutual":
        keep = best21[tgt, 0] == src
        src, tgt = src[keep], tgt[keep]
        score = (margin / (1.0 + cycle))[keep]
        order = np.argsort(-score)
        src, tgt, score = src[order], tgt[order], score[order]
    else:
        raise ValueError(f"Unknown method: {method}")

    return candidate_idx[src], candidate_idx[tgt], score.astype(np.float32)


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


def recall_fields(prefix, recall):
    if recall is None:
        nan = float("nan")
        return {
            f"{prefix}_t2i_R1": nan,
            f"{prefix}_t2i_R5": nan,
            f"{prefix}_t2i_R10": nan,
            f"{prefix}_i2t_R1": nan,
            f"{prefix}_i2t_R5": nan,
            f"{prefix}_i2t_R10": nan,
            f"{prefix}_mean_R10": nan,
        }
    return {
        f"{prefix}_t2i_R1": recall["t2i"]["R1"],
        f"{prefix}_t2i_R5": recall["t2i"]["R5"],
        f"{prefix}_t2i_R10": recall["t2i"]["R10"],
        f"{prefix}_i2t_R1": recall["i2t"]["R1"],
        f"{prefix}_i2t_R5": recall["i2t"]["R5"],
        f"{prefix}_i2t_R10": recall["i2t"]["R10"],
        f"{prefix}_mean_R10": 0.5 * (recall["t2i"]["R10"] + recall["i2t"]["R10"]),
    }


def metric_row(args, n_real, n_pseudo, method, pool_size, coverage,
               mean_conf, fm_diag, quality, direct, cca_recall, final):
    return {
        "dataset": args.data,
        "seed": args.seed,
        "n_real": n_real,
        "n_pseudo": n_pseudo,
        "method": method,
        "fm_ridge": args.fm_ridge,
        "pseudo_pool_size": pool_size,
        "pseudo_coverage": coverage,
        "mean_confidence": mean_conf,
        **fm_diag,
        **quality,
        **recall_fields("fm_direct", direct),
        **recall_fields("cca", cca_recall),
        **recall_fields("final", final),
    }


def evaluate(args, train1, train2, test1, test2, idx1, idx2,
             quality, direct, method, n_real, n_pseudo, pool_size,
             coverage, mean_conf, fm_diag, n_components, device):
    w1, w2 = fit_cross_cca(train1, train2, idx1, idx2, n_components)
    z = project(train1, train2, test1, test2, w1, w2)
    cca_recall = compute_bidirectional_recall(z["test1"], z["test2"])
    print_recall(f"[{n_real} real + {n_pseudo} pseudo | {method}] SE -> CCA", cca_recall)

    final = None
    if not args.skip_mmd:
        mmd = run_mmd(
            z["train1"], z["train2"], z["test1"], z["test2"],
            device=device, seed=args.seed, epochs=args.mmd_epochs,
            batch_size=args.mmd_batch_size, n_scales=args.mmd_scales,
        )
        final = compute_bidirectional_recall(mmd["test1"], mmd["test2"])
        print_recall(
            f"[{n_real} real + {n_pseudo} pseudo | {method}] SE -> CCA -> MMD",
            final,
        )

    row = metric_row(
        args, n_real, n_pseudo, method, pool_size, coverage, mean_conf,
        fm_diag, quality, direct, cca_recall, final,
    )
    return row, w1, w2


def self_check():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(80, 8)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    y = (x @ q).astype(np.float32)
    anchors = np.arange(40)
    c12, c21, diag = fit_fm(x, y, anchors, 1e-5)
    cand = np.arange(40, 80)
    pi, pt, _ = fm_matches(x, y, cand, c12, c21, "fm_mutual")
    assert len(pi) >= 35
    assert np.mean(pi == pt) > 0.95
    assert diag["fm_fit_error"] < 1e-3
    print("Step17 self-check passed.")


def parse_args():
    p = argparse.ArgumentParser(description="SUE Step17 - Functional Map replacement")
    p.add_argument("data", nargs="?", help="e.g. flickr30")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--pseudo_counts", type=int, nargs="+", default=[25, 50, 100])
    p.add_argument("--fm_ridge", type=float, default=1e-3)
    p.add_argument("--with_controls", action="store_true")
    p.add_argument("--control_method", choices=METHODS, default="fm_confidence")
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
    if args.fm_ridge < 0:
        p.error("fm_ridge must be >= 0")
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
            raise ValueError(f"n_real must be in [{n_components + 1}, {len(pair_order)}]")

    device = torch.device(
        args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
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

    candidate_idx = np.setdiff1d(np.arange(len(train1)), pair_order, assume_unique=True)
    rows = []
    rng = np.random.default_rng(args.seed + 17_000)

    print(f"Step17 | data={args.data} | seed={args.seed} | device={device}")
    print(f"pairs={args.pairs} | methods={args.methods} | pseudo={args.pseudo_counts}")
    print(f"fm_ridge={args.fm_ridge} | skip_mmd={args.skip_mmd}")
    print(f"unpaired candidates={len(candidate_idx)} | provenance=PASS")

    for n_real in args.pairs:
        print("\n" + "=" * 72)
        print(f"N_REAL = {n_real}")
        print("=" * 72)
        real_idx = pair_order[:n_real].copy()

        c12, c21, fm_diag = fit_fm(train1, train2, real_idx, args.fm_ridge)
        direct = compute_bidirectional_recall(test1 @ c12, test2)
        print_recall(f"[{n_real} real | anchor FM] direct mapping", direct)
        print(
            "FM diagnostics | "
            f"fit={fm_diag['fm_fit_error']:.6f} | "
            f"cycle={fm_diag['fm_cycle_error']:.6f} | "
            f"ortho={fm_diag['fm_ortho_error']:.6f}"
        )

        real_diag = {k: float("nan") for k in fm_diag}
        row, w1, w2 = evaluate(
            args, train1, train2, test1, test2,
            real_idx, real_idx, empty_quality(), None, "real_only",
            n_real, 0, 0, 0.0, float("nan"), real_diag,
            n_components, device,
        )
        rows.append(row)
        np.savez_compressed(
            diag_dir / f"real{n_real}_real_only.npz",
            real_indices=real_idx, projection1=w1, projection2=w2,
        )

        for method in args.methods:
            pseudo_image, pseudo_text, confidence = fm_matches(
                train1, train2, candidate_idx, c12, c21, method
            )
            pool_size = len(pseudo_image)
            coverage = pool_size / len(candidate_idx)
            print(
                f"[{method}] pool={pool_size} | coverage={coverage:.3f} | "
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
                q = pseudo_quality(pi, pt, candidate_idx, ids1, ids2, train1, train2)
                mean_conf = float(confidence[:n_pseudo].mean())
                row, w1, w2 = evaluate(
                    args, train1, train2, test1, test2,
                    np.r_[real_idx, pi], np.r_[real_idx, pt],
                    q, direct, method, n_real, n_pseudo, pool_size,
                    coverage, mean_conf, fm_diag, n_components, device,
                )
                rows.append(row)
                np.savez_compressed(
                    diag_dir / f"real{n_real}_{method}_pseudo{n_pseudo}.npz",
                    real_indices=real_idx,
                    pseudo_image_indices=pi,
                    pseudo_text_indices=pt,
                    confidence=confidence[:n_pseudo],
                    fm12=c12, fm21=c21,
                    projection1=w1, projection2=w2,
                )

                if args.with_controls and method == args.control_method:
                    shuffled = rng.permutation(pt)
                    q_shuf = pseudo_quality(
                        pi, shuffled, candidate_idx, ids1, ids2, train1, train2
                    )
                    row, _, _ = evaluate(
                        args, train1, train2, test1, test2,
                        np.r_[real_idx, pi], np.r_[real_idx, shuffled],
                        q_shuf, direct, f"{method}_shuffled", n_real, n_pseudo,
                        pool_size, coverage, mean_conf, fm_diag, n_components, device,
                    )
                    rows.append(row)

                    oi, ot = oracle_pairs(candidate_idx, ids1, ids2, n_pseudo, rng)
                    q_oracle = pseudo_quality(
                        oi, ot, candidate_idx, ids1, ids2, train1, train2
                    )
                    row, _, _ = evaluate(
                        args, train1, train2, test1, test2,
                        np.r_[real_idx, oi], np.r_[real_idx, ot],
                        q_oracle, direct, "oracle", n_real, n_pseudo,
                        pool_size, coverage, float("nan"), fm_diag,
                        n_components, device,
                    )
                    rows.append(row)

        save_results_csv(rows, run_dir / "results.csv")

    print(f"\nStep17 completed. Results: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
