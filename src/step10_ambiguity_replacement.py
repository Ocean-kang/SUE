"""
SUE Step 10 - Ambiguity Replacement.

New-file-only experiment:
- diagnose low-anchor ambiguity in Step 8 rank-signature matching;
- use anchor-subset consensus to keep structurally stable pseudo constraints;
- compare real-only / Step-9 relation / consensus relation / oracle.

CCA-only by design: Step 9 already isolated the replacement effect at CCA.
"""

import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from pair_removal import load_config, load_fixed_se_cache, set_seed
from step8_correspondence_replacement_v2 import (
    ROOT,
    empty_quality,
    fit_cca,
    mutual_matches,
    oracle_pairs,
    pseudo_quality,
    rank_signature,
    replay_weak_ids,
)

DEFAULT_OUT = ROOT / "results" / "step10_ambiguity"


def ambiguity_stats(image_sig, text_sig, topk, temperature, margin_threshold):
    def one_way(query, gallery):
        k = min(topk, len(gallery))
        dists = NearestNeighbors(n_neighbors=k, metric="cosine").fit(
            gallery
        ).kneighbors(query, return_distance=True)[0]

        relative_margin = (
            (dists[:, 1] - dists[:, 0]) / (dists[:, 0] + 1e-8)
            if k > 1
            else np.full(len(query), np.inf)
        )
        weights = np.exp(-(dists - dists[:, :1]) / temperature)
        probs = weights / weights.sum(axis=1, keepdims=True)
        entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        if k > 1:
            entropy /= np.log(k)

        return (
            float(np.mean(relative_margin)),
            float(np.median(relative_margin)),
            float(np.mean(entropy)),
            float(np.mean(relative_margin < margin_threshold)),
        )

    a = np.asarray(one_way(image_sig, text_sig))
    b = np.asarray(one_way(text_sig, image_sig))
    mean = (a + b) / 2
    return {
        "ambiguity_mean_margin": mean[0],
        "ambiguity_median_margin": mean[1],
        "ambiguity_entropy": mean[2],
        "ambiguity_fraction": mean[3],
    }


def consensus_matches(train1, train2, candidate_idx, real_idx, n_subsets,
                      subset_ratio, rng):
    n_subset = max(2, int(round(len(real_idx) * subset_ratio)))
    if n_subset >= len(real_idx):
        raise ValueError("subset_ratio must leave out at least one anchor.")

    votes = defaultdict(int)
    margin_sum = defaultdict(float)

    for _ in range(n_subsets):
        subset = rng.choice(real_idx, size=n_subset, replace=False)
        image_sig = rank_signature(train1[candidate_idx], train1[subset])
        text_sig = rank_signature(train2[candidate_idx], train2[subset])
        image_local, text_local, confidence = mutual_matches(image_sig, text_sig)

        for i, j, margin in zip(image_local, text_local, confidence):
            key = (int(i), int(j))
            votes[key] += 1
            margin_sum[key] += float(margin)

    records = []
    for (i, j), count in votes.items():
        consensus = count / n_subsets
        mean_margin = margin_sum[(i, j)] / count
        records.append({
            "image_local": i,
            "text_local": j,
            "votes": count,
            "consensus": consensus,
            "mean_margin": mean_margin,
            "score": consensus * max(mean_margin, 0.0),
        })
    records.sort(key=lambda r: (r["score"], r["consensus"]), reverse=True)
    return records


def select_one_to_one(records, n):
    # ponytail: greedy conflict removal is enough for 75 constraints.
    selected = []
    used_image, used_text = set(), set()
    for r in records:
        i, j = r["image_local"], r["text_local"]
        if i in used_image or j in used_text:
            continue
        selected.append(r)
        used_image.add(i)
        used_text.add(j)
        if len(selected) == n:
            break

    if len(selected) < n:
        raise ValueError(
            f"Only {len(selected)} one-to-one consensus pairs; requested {n}."
        )
    return selected


def recall_fields(recall):
    return {
        "cca_t2i_R1": recall["t2i"]["R1"],
        "cca_t2i_R5": recall["t2i"]["R5"],
        "cca_t2i_R10": recall["t2i"]["R10"],
        "cca_i2t_R1": recall["i2t"]["R1"],
        "cca_i2t_R5": recall["i2t"]["R5"],
        "cca_i2t_R10": recall["i2t"]["R10"],
        "cca_mean_R10": (
            recall["t2i"]["R10"] + recall["i2t"]["R10"]
        ) / 2,
    }


def make_row(condition, n_real, n_pseudo, quality, recall, ambiguity,
             mean_consensus="", mean_margin=""):
    return {
        "condition": condition,
        "n_real": n_real,
        "n_pseudo": n_pseudo,
        **quality,
        **ambiguity,
        "selected_mean_consensus": mean_consensus,
        "selected_mean_margin": mean_margin,
        **recall_fields(recall),
    }


def self_check():
    records = [
        {"image_local": 0, "text_local": 0, "score": 1.0, "consensus": 1.0},
        {"image_local": 0, "text_local": 1, "score": 0.9, "consensus": 0.9},
        {"image_local": 1, "text_local": 1, "score": 0.8, "consensus": 0.8},
    ]
    selected = select_one_to_one(records, 2)
    assert [(r["image_local"], r["text_local"]) for r in selected] == [
        (0, 0), (1, 1)
    ]
    print("Step 10 self-check: PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SUE Step 10 - Ambiguity Replacement"
    )
    parser.add_argument("data", nargs="?", default="flickr30")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_real", type=int, default=25)
    parser.add_argument("--n_pseudo", type=int, default=75)
    parser.add_argument("--subsets", type=int, default=10)
    parser.add_argument("--subset_ratio", type=float, default=0.8)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--ambiguity_margin", type=float, default=0.1)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--self_check", action="store_true")
    args = parser.parse_args()

    if not 0 < args.subset_ratio < 1:
        parser.error("--subset_ratio must be in (0, 1).")
    if args.subsets < 2:
        parser.error("--subsets must be >= 2.")
    if args.n_pseudo <= 0:
        parser.error("--n_pseudo must be > 0.")
    if args.topk < 2:
        parser.error("--topk must be >= 2.")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0.")
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

    if args.n_real <= n_components or args.n_real > len(pair_order):
        raise ValueError(
            f"n_real must be in [{n_components + 1}, {len(pair_order)}]."
        )

    real_idx = pair_order[:args.n_real].copy()
    candidate_idx = np.setdiff1d(
        np.arange(len(train1)), pair_order, assume_unique=True
    )

    encoded1 = torch.load(
        ROOT / "data" / args.data / "encoded1.pt",
        map_location="cpu",
    )
    n_original_train = len(encoded1) - int(config["n_test"])
    del encoded1

    ids1, ids2 = replay_weak_ids(
        n_original_train, len(pair_order), args.seed
    )
    if len(ids1) != len(train1) or len(ids2) != len(train2):
        raise RuntimeError("Weak-data replay != fixed-SE cache.")
    if not np.array_equal(ids1[pair_order], ids2[pair_order]):
        raise RuntimeError("Master-pair provenance replay failed.")

    print(
        f"Step10 | seed={args.seed} | real={args.n_real} | "
        f"pseudo={args.n_pseudo}"
    )

    image_sig = rank_signature(train1[candidate_idx], train1[real_idx])
    text_sig = rank_signature(train2[candidate_idx], train2[real_idx])
    ambiguity = ambiguity_stats(
        image_sig,
        text_sig,
        args.topk,
        args.temperature,
        args.ambiguity_margin,
    )

    image_local, text_local, confidence = mutual_matches(image_sig, text_sig)
    if len(image_local) < args.n_pseudo:
        raise ValueError(
            f"Only {len(image_local)} Step-9 relation pairs; "
            f"requested {args.n_pseudo}."
        )
    relation_i = candidate_idx[image_local[:args.n_pseudo]]
    relation_t = candidate_idx[text_local[:args.n_pseudo]]

    records = consensus_matches(
        train1,
        train2,
        candidate_idx,
        real_idx,
        args.subsets,
        args.subset_ratio,
        np.random.default_rng(args.seed + 10_000),
    )
    selected = select_one_to_one(records, args.n_pseudo)
    consensus_i = candidate_idx[
        np.array([r["image_local"] for r in selected], dtype=np.int64)
    ]
    consensus_t = candidate_idx[
        np.array([r["text_local"] for r in selected], dtype=np.int64)
    ]

    rows = []

    real_recall = fit_cca(
        train1, train2, test1, test2,
        real_idx, real_idx, n_components,
    )
    rows.append(make_row(
        "real_only", args.n_real, 0, empty_quality(), real_recall, ambiguity
    ))

    relation_quality = pseudo_quality(
        relation_i, relation_t, candidate_idx, ids1, ids2, train1, train2
    )
    relation_recall = fit_cca(
        train1, train2, test1, test2,
        np.r_[real_idx, relation_i],
        np.r_[real_idx, relation_t],
        n_components,
    )
    rows.append(make_row(
        f"relation_{args.n_pseudo}",
        args.n_real,
        args.n_pseudo,
        relation_quality,
        relation_recall,
        ambiguity,
        mean_margin=float(np.mean(confidence[:args.n_pseudo])),
    ))

    consensus_quality = pseudo_quality(
        consensus_i, consensus_t, candidate_idx, ids1, ids2, train1, train2
    )
    consensus_recall = fit_cca(
        train1, train2, test1, test2,
        np.r_[real_idx, consensus_i],
        np.r_[real_idx, consensus_t],
        n_components,
    )
    rows.append(make_row(
        f"consensus_{args.n_pseudo}",
        args.n_real,
        args.n_pseudo,
        consensus_quality,
        consensus_recall,
        ambiguity,
        mean_consensus=float(np.mean([r["consensus"] for r in selected])),
        mean_margin=float(np.mean([r["mean_margin"] for r in selected])),
    ))

    oracle_i, oracle_t = oracle_pairs(
        candidate_idx,
        ids1,
        ids2,
        args.n_pseudo,
        np.random.default_rng(args.seed),
    )
    oracle_quality = pseudo_quality(
        oracle_i, oracle_t, candidate_idx, ids1, ids2, train1, train2
    )
    oracle_recall = fit_cca(
        train1, train2, test1, test2,
        np.r_[real_idx, oracle_i],
        np.r_[real_idx, oracle_t],
        n_components,
    )
    rows.append(make_row(
        f"oracle_{args.n_pseudo}",
        args.n_real,
        args.n_pseudo,
        oracle_quality,
        oracle_recall,
        ambiguity,
    ))

    out_dir = (
        Path(args.output_dir)
        / args.data
        / f"seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"real{args.n_real}.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    selected_path = out_dir / f"real{args.n_real}_consensus_pairs.csv"
    with open(selected_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)

    print(
        f"Ambiguity: entropy={ambiguity['ambiguity_entropy']:.4f} | "
        f"fraction={ambiguity['ambiguity_fraction']:.4f} | "
        f"margin={ambiguity['ambiguity_mean_margin']:.4f}"
    )
    print()
    print(f"{'Condition':>16} | {'MedRank':>8} | {'Hit@100':>7} | {'Mean@10':>7}")
    print("-" * 50)
    for r in rows:
        med = "-" if r["median_gt_rank"] == "" else f"{float(r['median_gt_rank']):.1f}"
        hit = (
            "-"
            if r["semantic_hit_R100"] == ""
            else f"{100 * float(r['semantic_hit_R100']):.1f}%"
        )
        print(
            f"{r['condition']:>16} | {med:>8} | {hit:>7} | "
            f"{r['cca_mean_R10']:>7.2f}"
        )

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {selected_path}")


if __name__ == "__main__":
    main()
