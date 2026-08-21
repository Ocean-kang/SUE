import argparse
import csv
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
    set_seed,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "step8"


def rank_signature(x, anchors):
    sim = l2_normalize(x) @ l2_normalize(anchors).T
    ranks = np.argsort(np.argsort(sim, axis=1), axis=1).astype(np.float32)
    ranks -= ranks.mean(axis=1, keepdims=True)
    return l2_normalize(ranks)


def mutual_matches(image_sig, text_sig):
    # ponytail: brute cosine is enough for Flickr30K; use ANN only if the pool grows.
    text_nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(text_sig)
    dists, text_best2 = text_nn.kneighbors(image_sig)

    image_nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(image_sig)
    _, image_best = image_nn.kneighbors(text_sig)

    best_text = text_best2[:, 0]
    image_local = np.arange(len(image_sig))
    keep = image_best[best_text, 0] == image_local
    confidence = (dists[:, 1] - dists[:, 0])[keep]

    image_local = image_local[keep]
    text_local = best_text[keep]
    order = np.argsort(-confidence)
    return image_local[order], text_local[order], confidence[order]


def replay_weak_ids(n_train, n_parallel, seed):
    # ponytail: old fixed-SE caches have no sample ids. Replay current data.py
    # instead of changing/rebuilding them. If data.py changes, save ids in cache.
    torch.manual_seed(seed)
    n_unpaired = n_train - n_parallel
    n_remove = int(0.1 * n_unpaired)

    remove1 = torch.randperm(n_unpaired)[:n_remove].numpy()
    remove2 = torch.randperm(n_unpaired)[:n_remove].numpy()

    mask1 = np.ones(n_unpaired, dtype=bool)
    mask2 = np.ones(n_unpaired, dtype=bool)
    mask1[remove1] = False
    mask2[remove2] = False

    ids1 = np.arange(n_unpaired)[mask1]
    ids2 = np.arange(n_unpaired)[mask2]
    ids2 = ids2[torch.randperm(len(ids2)).numpy()]

    paired = np.arange(n_unpaired, n_train)
    return np.r_[ids1, paired], np.r_[ids2, paired]


def fit_cca(train1, train2, test1, test2, idx1, idx2, n_components):
    if len(idx1) != len(idx2) or len(idx1) <= n_components:
        raise ValueError("Invalid CCA pair count.")

    cca = CCA(n_components=n_components)
    cca.fit(train1[idx1], train2[idx2])
    return compute_bidirectional_recall(
        test1 @ cca.x_rotations_,
        test2 @ cca.y_rotations_,
    )


def oracle_pairs(candidate_idx, ids1, ids2, n, rng):
    text_by_id = {int(ids2[i]): int(i) for i in candidate_idx}
    pairs = [
        (int(i), text_by_id[int(ids1[i])])
        for i in candidate_idx
        if int(ids1[i]) in text_by_id
    ]
    if n > len(pairs):
        raise ValueError(f"Only {len(pairs)} oracle pairs available.")

    picked = rng.choice(len(pairs), size=n, replace=False)
    pairs = [pairs[i] for i in picked]
    return (
        np.array([p[0] for p in pairs], dtype=np.int64),
        np.array([p[1] for p in pairs], dtype=np.int64),
    )


def row(name, n_real, n_pseudo, precision, recall):
    t2i = recall["t2i"]["R10"]
    i2t = recall["i2t"]["R10"]
    return {
        "condition": name,
        "n_real": n_real,
        "n_pseudo": n_pseudo,
        "pseudo_precision": precision,
        "cca_t2i_R10": t2i,
        "cca_i2t_R10": i2t,
        "cca_mean_R10": (t2i + i2t) / 2,
    }


def self_check():
    anchors = np.eye(3, dtype=np.float32)
    x = np.array(
        [[1, 0.2, 0.1], [0.1, 1, 0.3], [0.2, 0.1, 1]],
        dtype=np.float32,
    )
    s = rank_signature(x, anchors)
    image_idx, text_idx, _ = mutual_matches(s, s.copy())
    assert len(image_idx) == 3
    assert np.array_equal(image_idx, text_idx)
    print("Step8 self-check: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", default="flickr30")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_real", type=int, default=100)
    parser.add_argument("--pseudo_counts", type=int, nargs="+", default=[25, 50, 75])
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--self_check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    warnings.filterwarnings("ignore", category=UserWarning)
    set_seed(args.seed)

    config = load_config(args.data)
    n_components = int(config["n_components"])
    cache = load_fixed_se_cache(args.data, args.seed, args.cache)

    train1, train2 = cache["train_se1"], cache["train_se2"]
    test1, test2 = cache["test_se1"], cache["test_se2"]
    pair_order = cache["pair_order"]

    if args.n_real <= n_components or args.n_real > len(pair_order):
        raise ValueError(
            f"n_real must be in [{n_components + 1}, {len(pair_order)}]."
        )

    real_idx = pair_order[: args.n_real].copy()

    # Exclude the whole 500-pair master block from pseudo candidates.
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
        raise RuntimeError(
            "Weak-data replay != fixed-SE cache. "
            "build_fixed_se.py/data.py changed."
        )
    if not np.array_equal(ids1[pair_order], ids2[pair_order]):
        raise RuntimeError("Master-pair provenance replay failed.")

    print(f"Real anchors        : {len(real_idx)}")
    print(f"Unpaired candidates : {len(candidate_idx)}")

    image_sig = rank_signature(train1[candidate_idx], train1[real_idx])
    text_sig = rank_signature(train2[candidate_idx], train2[real_idx])
    image_local, text_local, confidence = mutual_matches(image_sig, text_sig)

    pseudo_image = candidate_idx[image_local]
    pseudo_text = candidate_idx[text_local]
    print(f"Mutual pseudo pool  : {len(pseudo_image)}")

    if not len(pseudo_image):
        raise RuntimeError("No mutual pseudo correspondences found.")

    rows = []
    real_recall = fit_cca(
        train1, train2, test1, test2,
        real_idx, real_idx, n_components,
    )
    rows.append(row("real_only", args.n_real, 0, "", real_recall))

    rng = np.random.default_rng(args.seed)

    for n in args.pseudo_counts:
        if n <= 0 or n > len(pseudo_image):
            raise ValueError(
                f"n_pseudo={n}; available mutual pairs={len(pseudo_image)}."
            )

        pi = pseudo_image[:n]
        pt = pseudo_text[:n]
        precision = float(np.mean(ids1[pi] == ids2[pt]))

        recall = fit_cca(
            train1, train2, test1, test2,
            np.r_[real_idx, pi],
            np.r_[real_idx, pt],
            n_components,
        )
        rows.append(row(f"relation_{n}", args.n_real, n, precision, recall))

        shuffled = rng.permutation(pt)
        shuffled_precision = float(np.mean(ids1[pi] == ids2[shuffled]))
        recall = fit_cca(
            train1, train2, test1, test2,
            np.r_[real_idx, pi],
            np.r_[real_idx, shuffled],
            n_components,
        )
        rows.append(
            row(
                f"shuffled_{n}",
                args.n_real,
                n,
                shuffled_precision,
                recall,
            )
        )

        oi, ot = oracle_pairs(candidate_idx, ids1, ids2, n, rng)
        recall = fit_cca(
            train1, train2, test1, test2,
            np.r_[real_idx, oi],
            np.r_[real_idx, ot],
            n_components,
        )
        rows.append(row(f"oracle_{n}", args.n_real, n, 1.0, recall))

    out = (
        Path(args.output_dir)
        / args.data
        / f"seed{args.seed}"
        / f"real{args.n_real}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"{'Condition':>16} | {'Prec':>7} | "
        f"{'T2I@10':>7} | {'I2T@10':>7} | {'Mean@10':>7}"
    )
    print("-" * 58)
    for r in rows:
        p = r["pseudo_precision"]
        p = "-" if p == "" else f"{100 * p:.1f}%"
        print(
            f"{r['condition']:>16} | {p:>7} | "
            f"{r['cca_t2i_R10']:>7.2f} | "
            f"{r['cca_i2t_R10']:>7.2f} | "
            f"{r['cca_mean_R10']:>7.2f}"
        )

    print(f"\nSaved: {out}")
    print(f"Best mutual confidence: {confidence[0]:.6f}")


if __name__ == "__main__":
    main()
