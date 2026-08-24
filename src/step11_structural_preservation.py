import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cross_decomposition import CCA

from pair_removal import (
    compute_bidirectional_recall,
    load_config,
    load_fixed_se_cache,
    print_recall,
    run_mmd,
    save_results_csv,
    set_seed,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "step11"
DEFAULT_PAIRS = [500, 300, 200, 175, 150, 125, 100, 75, 50, 25, 10]


def pair_corr_loss(x, y, eps=1e-6):
    x = (x - x.mean(0)) / (x.std(0, unbiased=False) + eps)
    y = (y - y.mean(0)) / (y.std(0, unbiased=False) + eps)
    return 1.0 - (x * y).mean()


def structure_reg(original, aligned, temperature=0.1, eps=1e-12):
    """STRUCTURE-style one-level geometry preservation loss."""
    original = F.normalize(original, dim=-1)
    aligned = F.normalize(aligned, dim=-1)
    original = original - original.mean(0, keepdim=True)
    aligned = aligned - aligned.mean(0, keepdim=True)

    p = F.softmax((original @ original.T) / temperature, dim=-1)
    q = F.softmax((aligned @ aligned.T) / temperature, dim=-1)
    m = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div((q + eps).log(), m + eps, reduction="batchmean")
        + F.kl_div((p + eps).log(), m + eps, reduction="batchmean")
    )


def fit_cca(train1, train2, pair_idx, n_components):
    cca = CCA(n_components=n_components)
    cca.fit(train1[pair_idx], train2[pair_idx])
    return cca.x_rotations_.astype(np.float32), cca.y_rotations_.astype(np.float32)


def optimize_projections(
    train1,
    train2,
    pair_idx,
    w1_init,
    w2_init,
    structure_lambda,
    steps,
    lr,
    batch_size,
    temperature,
    warmup_steps,
    device,
    seed,
):
    x = torch.as_tensor(train1, dtype=torch.float32, device=device)
    y = torch.as_tensor(train2, dtype=torch.float32, device=device)
    idx = torch.as_tensor(pair_idx, dtype=torch.long, device=device)
    w1_0 = torch.as_tensor(w1_init, dtype=torch.float32, device=device)
    w2_0 = torch.as_tensor(w2_init, dtype=torch.float32, device=device)
    w1 = torch.nn.Parameter(w1_0.clone())
    w2 = torch.nn.Parameter(w2_0.clone())
    optimizer = torch.optim.Adam([w1, w2], lr=lr)
    rng = np.random.default_rng(seed)
    n = len(train1)
    b = min(batch_size, n)

    for step in range(steps):
        pair_loss = pair_corr_loss(x[idx] @ w1, y[idx] @ w2)

        ix = torch.as_tensor(rng.choice(n, b, replace=False), device=device)
        iy = torch.as_tensor(rng.choice(n, b, replace=False), device=device)
        struct_loss = 0.5 * (
            structure_reg(x[ix], x[ix] @ w1, temperature)
            + structure_reg(y[iy], y[iy] @ w2, temperature)
        )
        weight = structure_lambda
        if warmup_steps > 0:
            weight *= min(1.0, (step + 1) / warmup_steps)
        loss = pair_loss + weight * struct_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ponytail: both losses are scale-insensitive; keep CCA column scales fixed.
        with torch.no_grad():
            w1.mul_(w1_0.norm(dim=0, keepdim=True) / (w1.norm(dim=0, keepdim=True) + 1e-12))
            w2.mul_(w2_0.norm(dim=0, keepdim=True) / (w2.norm(dim=0, keepdim=True) + 1e-12))

        if step == 0 or step + 1 == steps or (step + 1) % 100 == 0:
            print(
                f"  step {step + 1:4d}/{steps} | "
                f"pair={pair_loss.item():.5f} | "
                f"struct={struct_loss.item():.5f} | lambda={weight:.4g}"
            )

    return w1.detach().cpu().numpy(), w2.detach().cpu().numpy()


def project(train1, train2, test1, test2, w1, w2):
    return {
        "train1": train1 @ w1,
        "train2": train2 @ w2,
        "test1": test1 @ w1,
        "test2": test2 @ w2,
    }


def geometry_js(original, aligned, sample_idx, temperature, device):
    with torch.no_grad():
        x = torch.as_tensor(original[sample_idx], dtype=torch.float32, device=device)
        z = torch.as_tensor(aligned[sample_idx], dtype=torch.float32, device=device)
        return float(structure_reg(x, z, temperature).item())


def metric_row(dataset, seed, n_pairs, condition, lam, projected, recall, final, img_js, txt_js):
    nan = float("nan")
    return {
        "dataset": dataset,
        "seed": seed,
        "n_pairs": n_pairs,
        "condition": condition,
        "structure_lambda": lam,
        "image_structure_js": img_js,
        "text_structure_js": txt_js,
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
        "final_mean_R10": 0.5 * (final["t2i"]["R10"] + final["i2t"]["R10"]) if final else nan,
    }


def self_check():
    torch.manual_seed(0)
    x = torch.randn(32, 8)
    same = structure_reg(x, x)
    moved = structure_reg(x, x.roll(1, 0))
    corr = pair_corr_loss(x, x)
    assert same.item() < 1e-6
    assert moved.item() > same.item()
    assert corr.item() < 1e-5
    print("Step11 self-check passed.")


def parse_args():
    p = argparse.ArgumentParser(description="SUE Step11 - STRUCTURE-preserved CCA")
    p.add_argument("data", nargs="?", help="e.g. flickr30")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--structure_lambda", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--structure_batch_size", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--diag_samples", type=int, default=256)
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
    if args.structure_lambda < 0 or args.steps <= 0 or args.lr <= 0:
        p.error("lambda must be >=0; steps/lr must be >0")
    if args.structure_batch_size < 2 or args.temperature <= 0:
        p.error("structure_batch_size must be >=2 and temperature >0")
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
    diag_rng = np.random.default_rng(args.seed + 11_000)
    diag_idx = diag_rng.choice(len(train1), min(args.diag_samples, len(train1)), replace=False)

    tag = str(args.structure_lambda).replace(".", "p")
    run_dir = Path(args.output_dir) / args.data / f"seed{args.seed}" / f"lambda_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"Step11 | data={args.data} | seed={args.seed} | device={device}")
    print(f"pairs={args.pairs} | lambda={args.structure_lambda} | skip_mmd={args.skip_mmd}")

    for n_pairs in args.pairs:
        print("\n" + "=" * 72)
        print(f"N_PAIRS = {n_pairs}")
        print("=" * 72)
        pair_idx = pair_order[:n_pairs].copy()
        w1_cca, w2_cca = fit_cca(train1, train2, pair_idx, n_components)

        print("[pair_ft] CCA-init + paired correlation control")
        w1_ft, w2_ft = optimize_projections(
            train1, train2, pair_idx, w1_cca, w2_cca,
            0.0, args.steps, args.lr, args.structure_batch_size,
            args.temperature, args.warmup_steps, device, args.seed + n_pairs,
        )

        print("[structure] CCA-init + paired correlation + STRUCTURE")
        w1_struct, w2_struct = optimize_projections(
            train1, train2, pair_idx, w1_cca, w2_cca,
            args.structure_lambda, args.steps, args.lr, args.structure_batch_size,
            args.temperature, args.warmup_steps, device, args.seed + n_pairs,
        )

        conditions = {
            "cca": (w1_cca, w2_cca, 0.0),
            "pair_ft": (w1_ft, w2_ft, 0.0),
            "structure": (w1_struct, w2_struct, args.structure_lambda),
        }

        for name, (w1, w2, lam) in conditions.items():
            projected = project(train1, train2, test1, test2, w1, w2)
            recall = compute_bidirectional_recall(projected["test1"], projected["test2"])
            print_recall(f"[{n_pairs} pairs | {name}] CCA-only", recall)

            img_js = geometry_js(train1, projected["train1"], diag_idx, args.temperature, device)
            txt_js = geometry_js(train2, projected["train2"], diag_idx, args.temperature, device)

            final = None
            if not args.skip_mmd:
                mmd = run_mmd(
                    projected["train1"], projected["train2"],
                    projected["test1"], projected["test2"],
                    device=device,
                    seed=args.seed,
                    epochs=args.mmd_epochs,
                    batch_size=args.mmd_batch_size,
                    n_scales=args.mmd_scales,
                )
                final = compute_bidirectional_recall(mmd["test1"], mmd["test2"])
                print_recall(f"[{n_pairs} pairs | {name}] CCA + MMD", final)

            rows.append(metric_row(
                args.data, args.seed, n_pairs, name, lam,
                projected, recall, final, img_js, txt_js,
            ))

            diag_dir = run_dir / "diagnostics"
            diag_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                diag_dir / f"pairs_{n_pairs}_{name}.npz",
                pair_indices=pair_idx,
                projection1=w1,
                projection2=w2,
            )

        save_results_csv(rows, run_dir / "results.csv")

    print(f"\nStep11 completed. Results: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
