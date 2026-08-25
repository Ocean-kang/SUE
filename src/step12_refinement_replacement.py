import argparse
import copy
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

from mmd import MMDLoss, MMDNet
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
from step11_structural_preservation import (
    fit_cca,
    geometry_js,
    optimize_projections,
    project,
)
from step8_correspondence_replacement_v2 import (
    empty_quality,
    oracle_pairs,
    pseudo_quality,
    replay_weak_ids,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "step12"
DEFAULT_PAIRS = [500, 300, 200, 175, 150, 125, 100, 75, 50, 25, 10]


def mutual_matches_shared(image_z, text_z):
    """Mutual NN in the current shared CCA space, ranked by NN margin."""
    image_z = l2_normalize(image_z)
    text_z = l2_normalize(text_z)
    text_nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(text_z)
    dists, best2 = text_nn.kneighbors(image_z)
    image_nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(image_z)
    _, back = image_nn.kneighbors(text_z)

    image_local = np.arange(len(image_z))
    best_text = best2[:, 0]
    keep = back[best_text, 0] == image_local
    confidence = (dists[:, 1] - dists[:, 0])[keep]
    order = np.argsort(-confidence)
    return image_local[keep][order], best_text[keep][order], confidence[order]


def stable_rng(seed, n_pairs, stream):
    mixed = (seed * 1_000_003 + n_pairs * 10_007 + stream * 97 + 17) % (2 ** 32)
    return np.random.default_rng(mixed)


def transform(model, x, device):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()


def pseudo_alignment_loss(model, x, y):
    return (1.0 - F.cosine_similarity(model(x), y, dim=-1)).mean()


def train_refiner(
    train1,
    train2,
    pseudo_i,
    pseudo_t,
    device,
    seed,
    epochs,
    batch_size,
    n_scales,
    pseudo_weight,
    displacement_weight,
    use_mmd,
):
    """Same MMDNet capacity; optionally add MMD for the full SUE stage."""
    set_seed(seed)
    x = torch.as_tensor(train1, dtype=torch.float32)
    y = torch.as_tensor(train2, dtype=torch.float32)
    pi = torch.as_tensor(train1[pseudo_i], dtype=torch.float32, device=device)
    pt = torch.as_tensor(train2[pseudo_t], dtype=torch.float32, device=device)

    model = MMDNet(train1.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mmd = MMDLoss(device, n_scales=n_scales) if use_mmd else None

    if use_mmd:
        n_val = int(0.1 * len(x))
        order = np.random.permutation(len(x))
        train_idx, val_idx = order[n_val:], order[:n_val]
        train_loader = DataLoader(
            TensorDataset(x[train_idx], y[train_idx]),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(x[val_idx], y[val_idx]),
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        train_loader = [None]
        val_loader = None

    best_model = None
    best_val = float("inf")
    rng = np.random.default_rng(seed + 123)
    disp_n = min(256, len(x))

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            pseudo = pseudo_alignment_loss(model, pi, pt)

            disp_idx = rng.choice(len(x), disp_n, replace=False)
            disp_x = x[disp_idx].to(device)
            displacement = F.mse_loss(model(disp_x), disp_x)

            loss = pseudo_weight * pseudo + displacement_weight * displacement
            if use_mmd:
                bx, by = batch
                bx, by = bx.to(device), by.to(device)
                loss = loss + mmd(model(bx), by)

            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        if use_mmd:
            model.eval()
            val = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    val += float(mmd(model(bx), by).item())
            val /= len(val_loader)
            if val < best_val:
                best_val = val
                best_model = copy.deepcopy(model)

        if epoch == 1 or epoch == epochs or epoch % 25 == 0:
            suffix = f" | val_mmd={best_val:.5f}" if use_mmd else ""
            print(f"  epoch {epoch:3d}/{epochs} | loss={epoch_loss / len(train_loader):.5f}{suffix}")

    return best_model if use_mmd else model


def mean_displacement(model, x, device, n=512):
    if model is None:
        return 0.0
    idx = np.arange(min(n, len(x)))
    before = torch.as_tensor(x[idx], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        after = model(before)
    return float(torch.linalg.vector_norm(after - before, dim=1).mean().item())


def metric_row(
    dataset,
    seed,
    n_pairs,
    condition,
    n_pseudo,
    pool_size,
    confidence,
    quality,
    cca_recall,
    final_recall,
    img_js,
    txt_js,
    displacement,
):
    nan = float("nan")
    conf = np.asarray(confidence, dtype=float)
    return {
        "dataset": dataset,
        "seed": seed,
        "n_pairs": n_pairs,
        "condition": condition,
        "n_pseudo": n_pseudo,
        "mutual_pool_size": pool_size,
        "confidence_mean": float(conf.mean()) if len(conf) else nan,
        "confidence_min": float(conf.min()) if len(conf) else nan,
        "confidence_max": float(conf.max()) if len(conf) else nan,
        **quality,
        "image_structure_js": img_js,
        "text_structure_js": txt_js,
        "mean_displacement": displacement,
        "cca_t2i_R1": cca_recall["t2i"]["R1"],
        "cca_t2i_R5": cca_recall["t2i"]["R5"],
        "cca_t2i_R10": cca_recall["t2i"]["R10"],
        "cca_i2t_R1": cca_recall["i2t"]["R1"],
        "cca_i2t_R5": cca_recall["i2t"]["R5"],
        "cca_i2t_R10": cca_recall["i2t"]["R10"],
        "cca_mean_R10": 0.5 * (cca_recall["t2i"]["R10"] + cca_recall["i2t"]["R10"]),
        "final_t2i_R1": final_recall["t2i"]["R1"] if final_recall else nan,
        "final_t2i_R5": final_recall["t2i"]["R5"] if final_recall else nan,
        "final_t2i_R10": final_recall["t2i"]["R10"] if final_recall else nan,
        "final_i2t_R1": final_recall["i2t"]["R1"] if final_recall else nan,
        "final_i2t_R5": final_recall["i2t"]["R5"] if final_recall else nan,
        "final_i2t_R10": final_recall["i2t"]["R10"] if final_recall else nan,
        "final_mean_R10": 0.5 * (final_recall["t2i"]["R10"] + final_recall["i2t"]["R10"]) if final_recall else nan,
    }


def self_check():
    x = np.eye(4, dtype=np.float32)
    i, t, c = mutual_matches_shared(x, x.copy())
    assert np.array_equal(i, t)
    assert len(c) == 4
    model = MMDNet(4)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    assert np.allclose(model(torch.from_numpy(x)).detach().numpy(), x)
    print("Step12 self-check passed.")


def parse_args():
    p = argparse.ArgumentParser(description="SUE Step12 - refinement replacement")
    p.add_argument("data", nargs="?", help="e.g. flickr30")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--n_pseudo", type=int, default=75)
    p.add_argument("--structure_lambda", type=float, default=1.0)
    p.add_argument("--structure_steps", type=int, default=500)
    p.add_argument("--structure_lr", type=float, default=1e-3)
    p.add_argument("--structure_batch_size", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--refine_epochs", type=int, default=100)
    p.add_argument("--pseudo_weight", type=float, default=1.0)
    p.add_argument("--displacement_weight", type=float, default=0.1)
    p.add_argument("--mmd_epochs", type=int, default=100)
    p.add_argument("--mmd_batch_size", type=int, default=32)
    p.add_argument("--mmd_scales", type=int, default=3)
    p.add_argument("--diag_samples", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip_mmd", action="store_true")
    p.add_argument("--cache", default=None)
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--self_check", action="store_true")
    args = p.parse_args()
    if not args.self_check and not args.data:
        p.error("data is required unless --self_check is used")
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

    for n_pairs in args.pairs:
        if n_pairs <= n_components or n_pairs > len(pair_order):
            raise ValueError(f"n_pairs must be in [{n_components + 1}, {len(pair_order)}]")

    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    candidate_idx = np.setdiff1d(np.arange(len(train1)), pair_order, assume_unique=True)

    encoded1 = torch.load(ROOT / "data" / args.data / "encoded1.pt", map_location="cpu")
    n_original_train = len(encoded1) - int(config["n_test"])
    del encoded1
    ids1, ids2 = replay_weak_ids(n_original_train, len(pair_order), args.seed)
    if len(ids1) != len(train1) or len(ids2) != len(train2):
        raise RuntimeError("Weak-data replay != fixed-SE cache.")
    if not np.array_equal(ids1[pair_order], ids2[pair_order]):
        raise RuntimeError("Master-pair provenance replay failed.")

    diag_rng = np.random.default_rng(args.seed + 12_000)
    diag_idx = diag_rng.choice(len(train1), min(args.diag_samples, len(train1)), replace=False)
    run_dir = Path(args.output_dir) / args.data / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"Step12 | data={args.data} | seed={args.seed} | device={device}")
    print(f"pairs={args.pairs} | n_pseudo={args.n_pseudo} | skip_mmd={args.skip_mmd}")
    print(f"pseudo candidates={len(candidate_idx)} | provenance=PASS")

    for n_pairs in args.pairs:
        print("\n" + "=" * 72)
        print(f"N_PAIRS = {n_pairs}")
        print("=" * 72)

        real_idx = pair_order[:n_pairs].copy()
        w1, w2 = fit_cca(train1, train2, real_idx, n_components)
        w1, w2 = optimize_projections(
            train1, train2, real_idx, w1, w2,
            args.structure_lambda, args.structure_steps, args.structure_lr,
            args.structure_batch_size, args.temperature, args.warmup_steps,
            device, args.seed + n_pairs,
        )
        base = project(train1, train2, test1, test2, w1, w2)

        image_local, text_local, confidence = mutual_matches_shared(
            base["train1"][candidate_idx], base["train2"][candidate_idx]
        )
        pseudo_i = candidate_idx[image_local]
        pseudo_t = candidate_idx[text_local]
        pool_size = len(pseudo_i)
        if pool_size < args.n_pseudo:
            raise RuntimeError(
                f"N_pairs={n_pairs}: only {pool_size} mutual matches, need {args.n_pseudo}."
            )

        rng_unfiltered = stable_rng(args.seed, n_pairs, 1)
        random_pick = rng_unfiltered.choice(pool_size, size=args.n_pseudo, replace=False)
        filtered_i = pseudo_i[:args.n_pseudo]
        filtered_t = pseudo_t[:args.n_pseudo]
        filtered_c = confidence[:args.n_pseudo]
        unfiltered_i = pseudo_i[random_pick]
        unfiltered_t = pseudo_t[random_pick]
        unfiltered_c = confidence[random_pick]
        shuffled_t = stable_rng(args.seed, n_pairs, 2).permutation(filtered_t)
        oracle_i, oracle_t = oracle_pairs(
            candidate_idx, ids1, ids2, args.n_pseudo, stable_rng(args.seed, n_pairs, 3)
        )

        conditions = {
            "unfiltered": (unfiltered_i, unfiltered_t, unfiltered_c),
            "filtered": (filtered_i, filtered_t, filtered_c),
            "shuffled": (filtered_i, shuffled_t, filtered_c),
            "oracle": (oracle_i, oracle_t, np.empty(0)),
        }

        base_recall = compute_bidirectional_recall(base["test1"], base["test2"])
        print_recall(f"[{n_pairs} pairs | structure] SE -> CCA", base_recall)
        base_final = None
        if not args.skip_mmd:
            mmd_result = run_mmd(
                base["train1"], base["train2"], base["test1"], base["test2"],
                device=device, seed=args.seed, epochs=args.mmd_epochs,
                batch_size=args.mmd_batch_size, n_scales=args.mmd_scales,
            )
            base_final = compute_bidirectional_recall(mmd_result["test1"], mmd_result["test2"])
            print_recall(f"[{n_pairs} pairs | structure] CCA + MMD", base_final)

        rows.append(metric_row(
            args.data, args.seed, n_pairs, "structure", 0, pool_size, [], empty_quality(),
            base_recall, base_final,
            geometry_js(train1, base["train1"], diag_idx, args.temperature, device),
            geometry_js(train2, base["train2"], diag_idx, args.temperature, device),
            0.0,
        ))

        for condition, (pi, pt, conf) in conditions.items():
            print(f"[{condition}] pseudo refinement")
            quality = pseudo_quality(pi, pt, candidate_idx, ids1, ids2, train1, train2)

            # CCA-stage experiment: pseudo refinement only, no MMD.
            refiner = train_refiner(
                base["train1"], base["train2"], pi, pt, device,
                seed=args.seed + n_pairs + 100,
                epochs=args.refine_epochs,
                batch_size=args.mmd_batch_size,
                n_scales=args.mmd_scales,
                pseudo_weight=args.pseudo_weight,
                displacement_weight=args.displacement_weight,
                use_mmd=False,
            )
            refined_train1 = transform(refiner, base["train1"], device)
            refined_test1 = transform(refiner, base["test1"], device)
            cca_recall = compute_bidirectional_recall(refined_test1, base["test2"])
            print_recall(f"[{n_pairs} pairs | {condition}] SE -> CCA + refinement", cca_recall)

            final_recall = None
            if not args.skip_mmd:
                # Full SUE experiment: same one-block MMDNet capacity as original MMD,
                # trained jointly with MMD + pseudo refinement + displacement.
                joint = train_refiner(
                    base["train1"], base["train2"], pi, pt, device,
                    seed=args.seed,
                    epochs=args.mmd_epochs,
                    batch_size=args.mmd_batch_size,
                    n_scales=args.mmd_scales,
                    pseudo_weight=args.pseudo_weight,
                    displacement_weight=args.displacement_weight,
                    use_mmd=True,
                )
                final_test1 = transform(joint, base["test1"], device)
                final_recall = compute_bidirectional_recall(final_test1, base["test2"])
                print_recall(f"[{n_pairs} pairs | {condition}] CCA + joint MMD refinement", final_recall)

            rows.append(metric_row(
                args.data, args.seed, n_pairs, condition, len(pi), pool_size, conf, quality,
                cca_recall, final_recall,
                geometry_js(train1, refined_train1, diag_idx, args.temperature, device),
                geometry_js(train2, base["train2"], diag_idx, args.temperature, device),
                mean_displacement(refiner, base["train1"], device),
            ))

        np.savez_compressed(
            run_dir / f"pairs_{n_pairs}_pseudo.npz",
            real_indices=real_idx,
            filtered_image=filtered_i,
            filtered_text=filtered_t,
            filtered_confidence=filtered_c,
        )
        save_results_csv(rows, run_dir / "results.csv")

    print(f"\nStep12 completed. Results: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
