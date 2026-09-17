import argparse
import json
import os
import random

import numpy as np
import torch

from data import load_dataset
from general_utils import load_checkpoint
from trainer import Trainer
from step21_fmot_core import (
    descriptor_coefficients,
    evaluate_retrieval,
    hks,
    joint_hks_times,
    make_strict_unpaired,
    prepare_basis,
    preprocess_test,
    refine_fm,
    solve_fm,
    spectral_transform,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_fixed_se(dataset, base_cfg, checkpoint, device):
    trainer = Trainer(
        dataset_name=dataset,
        n_parallel=base_cfg["n_parallel"],
        n_components=base_cfg["n_components"],
        configs=base_cfg,
        device=device,
    )
    load_checkpoint(trainer, checkpoint)
    trainer.spectralnet1.device = device
    trainer.spectralnet2.device = device
    trainer.spectralnet1.spec_net.to(device)
    trainer.spectralnet2.spec_net.to(device)
    return trainer.spectralnet1, trainer.spectralnet2


def state_path(dataset, mode, seed):
    return f"../checkpoints/step21_{dataset}_{mode}_seed{seed}.pth"


def result_path(dataset, mode, seed):
    os.makedirs("../results/step21", exist_ok=True)
    return f"../results/step21/{dataset}_{mode}_seed{seed}.json"


def print_result(mode, result):
    print(f"\n=== Step21 / {mode} ===")
    print("Image-to-text :", result["i2t"])
    print("Text-to-image :", result["t2i"])
    print(f"mR            : {result['mR']:.4f}")


def main():
    p = argparse.ArgumentParser("Step21: SUE + pair-free FM/OT")
    p.add_argument("data", choices=["flickr30", "mscoco"])
    p.add_argument("--mode", choices=["se", "fm", "fmot"], default="fmot")
    p.add_argument("--train", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--config", default="../configs/step21_fmot.json")
    p.add_argument("--se-checkpoint", default=None)
    args = p.parse_args()

    if not args.train and not args.test:
        args.train = True
        args.test = True

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base_cfg = load_json(f"../configs/{args.data}_config.json")
    cfg = load_json(args.config)
    se_ckpt = args.se_checkpoint or f"../checkpoints/checkpoints_{args.data}.pth"

    train_set, test_set = load_dataset(args.data, n_test=base_cfg["n_test"])
    sn_x, sn_y = load_fixed_se(args.data, base_cfg, se_ckpt, device)

    zx_test_raw = spectral_transform(sn_x, test_set[0])
    zy_test_raw = spectral_transform(sn_y, test_set[1])

    if args.mode == "se":
        result = evaluate_retrieval(zx_test_raw.to(device), zy_test_raw.to(device))
        print_result(args.mode, result)
        with open(result_path(args.data, args.mode, args.seed), "w") as f:
            json.dump(result, f, indent=2)
        return

    ckpt = state_path(args.data, args.mode, args.seed)

    if args.train:
        train_u = make_strict_unpaired(
            train_set,
            seed=args.seed,
            removal=cfg["removal_percentage"],
        )
        print(
            f"Strict unpaired: X={len(train_u[0])}, Y={len(train_u[1])}; "
            "no paired index is exposed to FM/OT."
        )

        zx = spectral_transform(sn_x, train_u[0])
        zy = spectral_transform(sn_y, train_u[1])

        m = min(cfg["support_size"], len(zx), len(zy))
        gx = torch.Generator().manual_seed(args.seed * 10 + 3)
        gy = torch.Generator().manual_seed(args.seed * 10 + 4)
        idx_x = torch.randperm(len(zx), generator=gx)[:m]
        idx_y = torch.randperm(len(zy), generator=gy)[:m]

        sx = prepare_basis(sn_x, train_u[0], zx, idx_x, device)
        sy = prepare_basis(sn_y, train_u[1], zy, idx_y, device)

        print(f"basis orth error: X={sx['orth_error']:.4f}, Y={sy['orth_error']:.4f}")

        times = joint_hks_times(sx["evals"], sy["evals"], cfg["n_hks"])
        hx = hks(sx["phi_sub"], sx["evals"], times)
        hy = hks(sy["phi_sub"], sy["evals"], times)

        A = descriptor_coefficients(sx["phi_sub"], hx)
        B = descriptor_coefficients(sy["phi_sub"], hy)

        Cxy = solve_fm(
            A, B, sx["evals"], sy["evals"],
            cfg["fm_init_lap"], cfg["fm_init_reg"]
        )
        Cyx = solve_fm(
            B, A, sy["evals"], sx["evals"],
            cfg["fm_init_lap"], cfg["fm_init_reg"]
        )

        if args.mode == "fmot":
            Cxy, Cyx = refine_fm(
                Cxy, Cyx,
                sx["phi_sub"], sy["phi_sub"],
                A, B, sx["evals"], sy["evals"], cfg
            )

        torch.save(
            {
                "Cxy": Cxy.cpu(),
                "Cyx": Cyx.cpu(),
                "x_scale": sx["scale"].cpu(),
                "y_scale": sy["scale"].cpu(),
                "x_perm": sx["perm"].cpu(),
                "y_perm": sy["perm"].cpu(),
                "x_evals": sx["evals"].cpu(),
                "y_evals": sy["evals"].cpu(),
                "mode": args.mode,
                "seed": args.seed,
                "se_checkpoint": se_ckpt,
            },
            ckpt,
        )
        print(f"saved: {ckpt}")

    if args.test:
        state = torch.load(ckpt, map_location=device)
        x_state = {
            "scale": state["x_scale"].to(device),
            "perm": state["x_perm"].to(device),
        }
        y_state = {
            "scale": state["y_scale"].to(device),
            "perm": state["y_perm"].to(device),
        }

        zx_test = preprocess_test(zx_test_raw, x_state, device)
        zy_test = preprocess_test(zy_test_raw, y_state, device)

        result = evaluate_retrieval(
            zx_test,
            zy_test,
            state["Cxy"].to(device),
            state["Cyx"].to(device),
        )
        print_result(args.mode, result)

        with open(result_path(args.data, args.mode, args.seed), "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
