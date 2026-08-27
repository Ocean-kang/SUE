"""SUE Step14 - contrastive alignment with structural preservation.

Compare CCA / Procrustes / Contrastive against structure-regularized
contrastive alignment under the same fixed-SE pair sweep, then optionally
run the unchanged SUE MMD refinement stage.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pair_removal import (
    compute_bidirectional_recall,
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
    project,
    structure_reg,
)
from step13_contrastive_procrustes import (
    alignment_stats,
    fit_procrustes,
    train_contrastive,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "step14"
DEFAULT_PAIRS = [500, 300, 200, 175, 150, 125, 100, 75, 50, 25, 10]
METHODS = (
    "cca",
    "procrustes",
    "contrastive",
    "contrastive_struct",
    "proc_cl_struct",
)


def train_contrastive_struct(
    train1,
    train2,
    pair_idx,
    n_components,
    device,
    seed,
    steps,
    lr,
    contrastive_temperature,
    structure_lambda,
    structure_temperature,
    structure_batch_size,
    warmup_steps,
    init=None,
):
    """Real-pair InfoNCE + unpaired intra-modal structure preservation."""
    set_seed(seed)
    x = torch.as_tensor(train1, dtype=torch.float32, device=device)
    y = torch.as_tensor(train2, dtype=torch.float32, device=device)
    idx = torch.as_tensor(pair_idx, dtype=torch.long, device=device)

    if init is None:
        w1 = torch.nn.Parameter(torch.empty(train1.shape[1], n_components, device=device))
        w2 = torch.nn.Parameter(torch.empty(train2.shape[1], n_components, device=device))
        torch.nn.init.orthogonal_(w1)
        torch.nn.init.orthogonal_(w2)
    else:
        w1 = torch.nn.Parameter(torch.as_tensor(init[0], dtype=torch.float32, device=device).clone())
        w2 = torch.nn.Parameter(torch.as_tensor(init[1], dtype=torch.float32, device=device).clone())

    optimizer = torch.optim.Adam([w1, w2], lr=lr)
    labels = torch.arange(len(pair_idx), device=device)
    rng = np.random.default_rng(seed)
    n = len(train1)
    b = min(structure_batch_size, n)

    for step in range(steps):
        z1 = F.normalize(x[idx] @ w1, dim=-1)
        z2 = F.normalize(y[idx] @ w2, dim=-1)
        logits = z1 @ z2.T / contrastive_temperature
        pair_loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )

        ix = torch.as_tensor(rng.choice(n, b, replace=False), device=device)
        iy = torch.as_tensor(rng.choice(n, b, replace=False), device=device)
        struct_loss = 0.5 * (
            structure_reg(x[ix], x[ix] @ w1, structure_temperature)
            + structure_reg(y[iy], y[iy] @ w2, structure_temperature)
        )

        weight = structure_lambda
        if warmup_steps > 0:
            weight *= min(1.0, (step + 1) / warmup_steps)

        loss = pair_loss + weight * struct_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Match Step13: InfoNCE is scale-invariant, keep column scale stable.
        with torch.no_grad():
            w1.div_(w1.norm(dim=0, keepdim=True).clamp_min(1e-12))
            w2.div_(w2.norm(dim=0, keepdim=True).clamp_min(1e-12))

        if step == 0 or step + 1 == steps or (step + 1) % 100 == 0:
            print(
                f" step {step + 1:4d}/{steps} | "
                f"InfoNCE={pair_loss.item():.5f} | "
                f"struct={struct_loss.item():.5f} | lambda={weight:.4g}"
            )

    return w1.detach().cpu().numpy(), w2.detach().cpu().numpy()


def metric_row(
    dataset,
    seed,
    n_pairs,
    method,
    structure_lambda,
    recall,
    final,
    pair_cosine,
    pair_margin,
    image_js,
    text_js,
    post_mmd_image_js,
    post_mmd_text_js,
):
    nan = float("nan")
    return {
        "dataset": dataset,
        "seed": seed,
        "n_pairs": n_pairs,
        "method": method,
        "structure_lambda": structure_lambda,
        "pair_cosine": pair_cosine,
        "pair_hard_margin": pair_margin,
        "image_structure_js": image_js,
        "text_structure_js": text_js,
        "align_t2i_R1": recall["t2i"]["R1"],
        "align_t2i_R5": recall["t2i"]["R5"],
        "align_t2i_R10": recall["t2i"]["R10"],
        "align_i2t_R1": recall["i2t"]["R1"],
        "align_i2t_R5": recall["i2t"]["R5"],
        "align_i2t_R10": recall["i2t"]["R10"],
        "align_mean_R10": 0.5 * (recall["t2i"]["R10"] + recall["i2t"]["R10"]),
        "post_mmd_image_structure_js": post_mmd_image_js,
        "post_mmd_text_structure_js": post_mmd_text_js,
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


def self_check():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 6)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    y = (x @ q).astype(np.float32)
    idx = np.arange(len(x))

    init = fit_procrustes(x, y, idx, 6)
    w1, w2 = train_contrastive_struct(
        x,
        y,
        idx,
        6,
        torch.device("cpu"),
        seed=0,
        steps=200,
        lr=1e-2,
        contrastive_temperature=0.07,
        structure_lambda=1.0,
        structure_temperature=0.1,
        structure_batch_size=32,
        warmup_steps=50,
        init=init,
    )
    cosine, _ = alignment_stats(x @ w1, y @ w2)
    assert cosine > 0.9
    print("Step14 self-check passed.")


def parse_args():
    p = argparse.ArgumentParser(
        description="SUE Step14 - Contrastive/Procrustes + structural preservation"
    )
    p.add_argument("data", nargs="?", help="e.g. flickr30")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)

    p.add_argument("--contrastive_steps", type=int, default=500)
    p.add_argument("--contrastive_lr", type=float, default=1e-2)
    p.add_argument("--contrastive_temperature", type=float, default=0.07)

    p.add_argument("--structure_lambda", type=float, default=1.0)
    p.add_argument("--structure_temperature", type=float, default=0.1)
    p.add_argument("--structure_batch_size", type=int, default=256)
    p.add_argument("--structure_warmup_steps", type=int, default=100)

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
    if args.contrastive_temperature <= 0 or args.structure_temperature <= 0:
        p.error("temperatures must be > 0")
    if args.structure_lambda < 0 or args.structure_batch_size <= 0:
        p.error("structure_lambda must be >= 0 and batch_size must be > 0")
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
    print(f"Step14 | data={args.data} | seed={args.seed} | device={device}")
    print(f"pairs={args.pairs}")
    print(f"methods={args.methods} | skip_mmd={args.skip_mmd}")
    print(
        f"structure_lambda={args.structure_lambda} | "
        f"structure_temperature={args.structure_temperature}"
    )

    geometry_rng = np.random.default_rng(args.seed + 14_000)
    geometry_n = min(args.structure_batch_size, len(train1))
    geometry_idx = geometry_rng.choice(len(train1), geometry_n, replace=False)

    for n_pairs in args.pairs:
        print("\n" + "=" * 72)
        print(f"N_PAIRS = {n_pairs}")
        print("=" * 72)
        pair_idx = pair_order[:n_pairs].copy()
        projections = {}

        if "cca" in args.methods:
            projections["cca"] = fit_cca(train1, train2, pair_idx, n_components)

        proc_init = None
        if "procrustes" in args.methods or "proc_cl_struct" in args.methods:
            proc_init = fit_procrustes(train1, train2, pair_idx, n_components)
            if "procrustes" in args.methods:
                projections["procrustes"] = proc_init

        if "contrastive" in args.methods:
            print("[contrastive] Step13 real-pair InfoNCE")
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

        if "contrastive_struct" in args.methods:
            print("[contrastive_struct] InfoNCE + unpaired STRUCTURE")
            projections["contrastive_struct"] = train_contrastive_struct(
                train1,
                train2,
                pair_idx,
                n_components,
                device,
                seed=args.seed + 14_000 + n_pairs,
                steps=args.contrastive_steps,
                lr=args.contrastive_lr,
                contrastive_temperature=args.contrastive_temperature,
                structure_lambda=args.structure_lambda,
                structure_temperature=args.structure_temperature,
                structure_batch_size=args.structure_batch_size,
                warmup_steps=args.structure_warmup_steps,
            )

        if "proc_cl_struct" in args.methods:
            print("[proc_cl_struct] Procrustes init -> InfoNCE + STRUCTURE")
            projections["proc_cl_struct"] = train_contrastive_struct(
                train1,
                train2,
                pair_idx,
                n_components,
                device,
                seed=args.seed + 14_500 + n_pairs,
                steps=args.contrastive_steps,
                lr=args.contrastive_lr,
                contrastive_temperature=args.contrastive_temperature,
                structure_lambda=args.structure_lambda,
                structure_temperature=args.structure_temperature,
                structure_batch_size=args.structure_batch_size,
                warmup_steps=args.structure_warmup_steps,
                init=proc_init,
            )

        for method, (w1, w2) in projections.items():
            projected = project(train1, train2, test1, test2, w1, w2)
            recall = compute_bidirectional_recall(projected["test1"], projected["test2"])
            print_recall(f"[{n_pairs} pairs | {method}] SE -> pair alignment", recall)

            pair_cosine, pair_margin = alignment_stats(
                projected["train1"][pair_idx], projected["train2"][pair_idx]
            )
            image_js = geometry_js(
                train1,
                projected["train1"],
                geometry_idx,
                args.structure_temperature,
                device,
            )
            text_js = geometry_js(
                train2,
                projected["train2"],
                geometry_idx,
                args.structure_temperature,
                device,
            )
            print(
                f"pair cosine={pair_cosine:.4f} | hard margin={pair_margin:.4f} | "
                f"structure_js=({image_js:.5f}, {text_js:.5f})"
            )

            final = None
            post_image_js = float("nan")
            post_text_js = float("nan")
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
                final = compute_bidirectional_recall(mmd["test1"], mmd["test2"])
                print_recall(
                    f"[{n_pairs} pairs | {method}] SE -> pair alignment -> MMD",
                    final,
                )
                post_image_js = geometry_js(
                    train1,
                    mmd["train1"],
                    geometry_idx,
                    args.structure_temperature,
                    device,
                )
                post_text_js = geometry_js(
                    train2,
                    mmd["train2"],
                    geometry_idx,
                    args.structure_temperature,
                    device,
                )

            rows.append(
                metric_row(
                    args.data,
                    args.seed,
                    n_pairs,
                    method,
                    args.structure_lambda,
                    recall,
                    final,
                    pair_cosine,
                    pair_margin,
                    image_js,
                    text_js,
                    post_image_js,
                    post_text_js,
                )
            )
            np.savez_compressed(
                diag_dir / f"pairs_{n_pairs}_{method}.npz",
                pair_indices=pair_idx,
                projection1=w1,
                projection2=w2,
            )

        save_results_csv(rows, run_dir / "results.csv")

    print(f"\nStep14 completed. Results: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
