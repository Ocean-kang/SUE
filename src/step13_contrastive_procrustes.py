"""SUE Step13 - pair-efficient alignment objectives.

Compare CCA / reduced two-sided Procrustes / linear contrastive alignment
under the same fixed-SE pair sweep, then run the same SUE MMD stage.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
from step11_structural_preservation import fit_cca, project

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "step13"
DEFAULT_PAIRS = [500, 300, 200, 175, 150, 125, 100, 75, 50, 25, 10]
METHODS = ("cca", "procrustes", "contrastive")


def fit_procrustes(train1, train2, pair_idx, n_components):
    """Reduced two-sided orthogonal Procrustes: SE -> shared k-D space."""
    x = train1[pair_idx].astype(np.float64)
    y = train2[pair_idx].astype(np.float64)
    x -= x.mean(0, keepdims=True)
    y -= y.mean(0, keepdims=True)
    u, _, vh = np.linalg.svd(x.T @ y, full_matrices=False)
    return (
        u[:, :n_components].astype(np.float32),
        vh.T[:, :n_components].astype(np.float32),
    )


def train_contrastive(
    train1,
    train2,
    pair_idx,
    n_components,
    device,
    seed,
    steps,
    lr,
    temperature,
):
    """Two bias-free linear projectors trained only on real pairs."""
    set_seed(seed)
    x = torch.as_tensor(train1[pair_idx], dtype=torch.float32, device=device)
    y = torch.as_tensor(train2[pair_idx], dtype=torch.float32, device=device)
    w1 = torch.nn.Parameter(
        torch.empty(train1.shape[1], n_components, device=device)
    )
    w2 = torch.nn.Parameter(
        torch.empty(train2.shape[1], n_components, device=device)
    )
    torch.nn.init.orthogonal_(w1)
    torch.nn.init.orthogonal_(w2)
    optimizer = torch.optim.Adam([w1, w2], lr=lr)
    labels = torch.arange(len(pair_idx), device=device)

    for step in range(steps):
        z1 = F.normalize(x @ w1, dim=-1)
        z2 = F.normalize(y @ w2, dim=-1)
        logits = z1 @ z2.T / temperature
        loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ponytail: InfoNCE is scale-invariant; fix column scales so MMD sees
        # a reproducible input scale instead of optimizer-dependent norms.
        with torch.no_grad():
            w1.div_(w1.norm(dim=0, keepdim=True).clamp_min(1e-12))
            w2.div_(w2.norm(dim=0, keepdim=True).clamp_min(1e-12))

        if step == 0 or step + 1 == steps or (step + 1) % 100 == 0:
            print(
                f"  step {step + 1:4d}/{steps} | "
                f"InfoNCE={loss.item():.5f}"
            )

    return w1.detach().cpu().numpy(), w2.detach().cpu().numpy()


def alignment_stats(image_z, text_z):
    """Positive cosine and positive-vs-hard-negative margin on real pairs."""
    x = l2_normalize(image_z)
    y = l2_normalize(text_z)
    sim = x @ y.T
    positive = np.diag(sim)
    if len(sim) == 1:
        margin = np.array([np.nan])
    else:
        negative = sim.copy()
        np.fill_diagonal(negative, -np.inf)
        margin = positive - negative.max(axis=1)
    return float(positive.mean()), float(np.nanmean(margin))


def metric_row(dataset, seed, n_pairs, method, recall, final, pair_cosine, pair_margin):
    nan = float("nan")
    return {
        "dataset": dataset,
        "seed": seed,
        "n_pairs": n_pairs,
        "method": method,
        "pair_cosine": pair_cosine,
        "pair_hard_margin": pair_margin,
        "align_t2i_R1": recall["t2i"]["R1"],
        "align_t2i_R5": recall["t2i"]["R5"],
        "align_t2i_R10": recall["t2i"]["R10"],
        "align_i2t_R1": recall["i2t"]["R1"],
        "align_i2t_R5": recall["i2t"]["R5"],
        "align_i2t_R10": recall["i2t"]["R10"],
        "align_mean_R10": 0.5 * (
            recall["t2i"]["R10"] + recall["i2t"]["R10"]
        ),
        "final_t2i_R1": final["t2i"]["R1"] if final else nan,
        "final_t2i_R5": final["t2i"]["R5"] if final else nan,
        "final_t2i_R10": final["t2i"]["R10"] if final else nan,
        "final_i2t_R1": final["i2t"]["R1"] if final else nan,
        "final_i2t_R5": final["i2t"]["R5"] if final else nan,
        "final_i2t_R10": final["i2t"]["R10"] if final else nan,
        "final_mean_R10": 0.5 * (
            final["t2i"]["R10"] + final["i2t"]["R10"]
        ) if final else nan,
    }


def self_check():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 6)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    y = (x @ q).astype(np.float32)
    idx = np.arange(len(x))

    w1, w2 = fit_procrustes(x, y, idx, 6)
    p1, p2 = x @ w1, y @ w2
    proc_cos, _ = alignment_stats(p1, p2)
    assert proc_cos > 0.999

    w1, w2 = train_contrastive(
        x, y, idx, 6, torch.device("cpu"),
        seed=0, steps=200, lr=1e-2, temperature=0.07,
    )
    c1, c2 = x @ w1, y @ w2
    con_cos, _ = alignment_stats(c1, c2)
    assert con_cos > 0.95
    print("Step13 self-check passed.")


def parse_args():
    p = argparse.ArgumentParser(
        description="SUE Step13 - CCA vs Procrustes vs Contrastive"
    )
    p.add_argument("data", nargs="?", help="e.g. flickr30")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--contrastive_steps", type=int, default=500)
    p.add_argument("--contrastive_lr", type=float, default=1e-2)
    p.add_argument("--contrastive_temperature", type=float, default=0.07)
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
    if args.contrastive_steps <= 0 or args.contrastive_lr <= 0:
        p.error("contrastive_steps/lr must be > 0")
    if args.contrastive_temperature <= 0:
        p.error("contrastive_temperature must be > 0")
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
            raise ValueError(
                f"n_pairs must be in [{n_components + 1}, {len(pair_order)}]"
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
    rows = []

    print(f"Step13 | data={args.data} | seed={args.seed} | device={device}")
    print(f"pairs={args.pairs}")
    print(f"methods={args.methods} | skip_mmd={args.skip_mmd}")

    for n_pairs in args.pairs:
        print("\n" + "=" * 72)
        print(f"N_PAIRS = {n_pairs}")
        print("=" * 72)
        pair_idx = pair_order[:n_pairs].copy()

        projections = {}
        if "cca" in args.methods:
            projections["cca"] = fit_cca(
                train1, train2, pair_idx, n_components
            )
        if "procrustes" in args.methods:
            projections["procrustes"] = fit_procrustes(
                train1, train2, pair_idx, n_components
            )
        if "contrastive" in args.methods:
            print("[contrastive] training real-pair InfoNCE")
            projections["contrastive"] = train_contrastive(
                train1,
                train2,
                pair_idx,
                n_components,
                device,
                seed=args.seed + 13_000 + n_pairs,
                steps=args.contrastive_steps,
                lr=args.contrastive_lr,
                temperature=args.contrastive_temperature,
            )

        for method, (w1, w2) in projections.items():
            projected = project(train1, train2, test1, test2, w1, w2)
            recall = compute_bidirectional_recall(
                projected["test1"], projected["test2"]
            )
            print_recall(
                f"[{n_pairs} pairs | {method}] SE -> alignment",
                recall,
            )

            pair_cosine, pair_margin = alignment_stats(
                projected["train1"][pair_idx],
                projected["train2"][pair_idx],
            )
            print(
                f"pair cosine={pair_cosine:.4f} | "
                f"hard margin={pair_margin:.4f}"
            )

            final = None
            if not args.skip_mmd:
                mmd = run_mmd(
                    projected["train1"],
                    projected["train2"],
                    projected["test1"],
                    projected["test2"],
                    device=device,
                    seed=args.seed,
                    epochs=args.mmd_epochs,
                    batch_size=args.mmd_batch_size,
                    n_scales=args.mmd_scales,
                )
                final = compute_bidirectional_recall(
                    mmd["test1"], mmd["test2"]
                )
                print_recall(
                    f"[{n_pairs} pairs | {method}] SE -> alignment -> MMD",
                    final,
                )

            rows.append(
                metric_row(
                    args.data,
                    args.seed,
                    n_pairs,
                    method,
                    recall,
                    final,
                    pair_cosine,
                    pair_margin,
                )
            )
            np.savez_compressed(
                diag_dir / f"pairs_{n_pairs}_{method}.npz",
                pair_indices=pair_idx,
                projection1=w1,
                projection2=w2,
            )

        save_results_csv(rows, run_dir / "results.csv")

    print(f"\nStep13 completed. Results: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
