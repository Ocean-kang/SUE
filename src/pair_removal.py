"""
pair_removal.py

Controlled Pair Removal experiment for SUE.

Goal
----
Keep SE embeddings fixed and only vary the number of real paired samples
used by CCA.

Pipeline
--------
Fixed SE
    ↓
Nested real pairs
    ↓
CCA
    ↓
CCA-only Recall
    ↓
MMD refinement
    ↓
CCA+MMD Recall

Expected fixed-SE cache
-----------------------
artifacts/fixed_se/flickr30_seed0.npz

Required keys:
    train_se1
    train_se2
    test_se1
    test_se2
    pair_order

pair_order contains indices into train_se1/train_se2 corresponding to
the 500 real paired samples in randomized nested order.

Example
-------
python pair_removal.py flickr30 \
    --seed 0 \
    --pairs 500 300 200 100 50 20 10

Single sanity check:
python pair_removal.py flickr30 \
    --seed 0 \
    --pairs 500
"""

import argparse
import csv
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch

from sklearn.cross_decomposition import CCA

from general_utils import calc_recall
from mmd import fine_tune_alignment_using_mmd_network


# ============================================================
# Paths
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DEFAULT_CACHE_DIR = PROJECT_ROOT / "artifacts" / "fixed_se"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "pair_removal"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    """
    Set random seeds for Python, NumPy and PyTorch.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Config
# ============================================================

def load_config(dataset_name: str):
    """
    Load SUE config file.
    """

    config_path = (
        PROJECT_ROOT
        / "configs"
        / f"{dataset_name}_config.json"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found:\n{config_path}"
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


# ============================================================
# Fixed SE cache
# ============================================================

def load_fixed_se_cache(
    dataset_name: str,
    seed: int,
    cache_path=None,
):
    """
    Load fixed Spectral Embedding outputs.

    Required arrays:
        train_se1
        train_se2
        test_se1
        test_se2
        pair_order
    """

    if cache_path is None:
        cache_path = (
            DEFAULT_CACHE_DIR
            / f"{dataset_name}_seed{seed}.npz"
        )
    else:
        cache_path = Path(cache_path)

    if not cache_path.exists():
        raise FileNotFoundError(
            "\nFixed SE cache does not exist:\n"
            f"{cache_path}\n\n"
            "Run build_fixed_se.py first."
        )

    print(f"Loading fixed SE cache:")
    print(f"  {cache_path}")

    data = np.load(cache_path, allow_pickle=False)

    required_keys = [
        "train_se1",
        "train_se2",
        "test_se1",
        "test_se2",
        "pair_order",
    ]

    missing = [
        key
        for key in required_keys
        if key not in data.files
    ]

    if missing:
        raise KeyError(
            f"Missing keys in fixed-SE cache: {missing}\n"
            f"Existing keys: {data.files}"
        )

    train_se1 = np.asarray(data["train_se1"])
    train_se2 = np.asarray(data["train_se2"])

    test_se1 = np.asarray(data["test_se1"])
    test_se2 = np.asarray(data["test_se2"])

    pair_order = np.asarray(
        data["pair_order"],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if train_se1.ndim != 2:
        raise ValueError(
            f"train_se1 must be 2D, got {train_se1.shape}"
        )

    if train_se2.ndim != 2:
        raise ValueError(
            f"train_se2 must be 2D, got {train_se2.shape}"
        )

    if test_se1.ndim != 2:
        raise ValueError(
            f"test_se1 must be 2D, got {test_se1.shape}"
        )

    if test_se2.ndim != 2:
        raise ValueError(
            f"test_se2 must be 2D, got {test_se2.shape}"
        )

    if len(train_se1) != len(train_se2):
        raise ValueError(
            "train_se1 and train_se2 "
            "must contain the same number of samples."
        )

    if len(test_se1) != len(test_se2):
        raise ValueError(
            "test_se1 and test_se2 "
            "must contain the same number of samples."
        )

    if pair_order.ndim != 1:
        raise ValueError(
            f"pair_order must be 1D, got {pair_order.shape}"
        )

    if len(pair_order) == 0:
        raise ValueError(
            "pair_order is empty."
        )

    if pair_order.min() < 0:
        raise ValueError(
            "pair_order contains negative indices."
        )

    if pair_order.max() >= len(train_se1):
        raise ValueError(
            "pair_order contains an index outside "
            "the training SE arrays."
        )

    if len(np.unique(pair_order)) != len(pair_order):
        raise ValueError(
            "pair_order contains duplicated indices."
        )

    print()
    print("Fixed SE cache summary:")
    print(f"  train_se1 : {train_se1.shape}")
    print(f"  train_se2 : {train_se2.shape}")
    print(f"  test_se1  : {test_se1.shape}")
    print(f"  test_se2  : {test_se2.shape}")
    print(f"  pair pool : {len(pair_order)}")
    print()

    return {
        "train_se1": train_se1,
        "train_se2": train_se2,
        "test_se1": test_se1,
        "test_se2": test_se2,
        "pair_order": pair_order,
        "cache_path": cache_path,
    }


# ============================================================
# Recall
# ============================================================

def l2_normalize(x: np.ndarray):
    """
    Row-wise L2 normalization.
    """

    norm = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    norm = np.maximum(norm, 1e-12)

    return x / norm


def recall_from_similarity(
    similarity: np.ndarray,
):
    """
    Calculate R@1 / R@5 / R@10 using SUE's calc_recall.
    """

    similarity_tensor = torch.from_numpy(
        similarity
    )

    r1, r5, r10 = calc_recall(
        similarity_tensor,
        labels=None,
    )

    return {
        "R1": float(r1.item()),
        "R5": float(r5.item()),
        "R10": float(r10.item()),
    }


def compute_bidirectional_recall(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
):
    """
    Same retrieval convention as SUE Trainer.

    Text -> Image:
        text @ image.T

    Image -> Text:
        image @ text.T
    """

    image_embeddings = l2_normalize(
        image_embeddings
    )

    text_embeddings = l2_normalize(
        text_embeddings
    )

    # Text -> Image
    similarity_t2i = (
        text_embeddings
        @ image_embeddings.T
    )

    t2i = recall_from_similarity(
        similarity_t2i
    )

    # Image -> Text
    similarity_i2t = (
        image_embeddings
        @ text_embeddings.T
    )

    i2t = recall_from_similarity(
        similarity_i2t
    )

    return {
        "t2i": t2i,
        "i2t": i2t,
    }


def print_recall(
    title: str,
    results: dict,
):
    """
    Print recall in the same style as SUE.
    """

    print(title)

    print("Text-to-image Recall:")
    print(
        f"Recall@1: "
        f"{results['t2i']['R1']:.2f}"
    )
    print(
        f"Recall@5: "
        f"{results['t2i']['R5']:.2f}"
    )
    print(
        f"Recall@10: "
        f"{results['t2i']['R10']:.2f}"
    )

    print("-" * 20)

    print("Image-to-text Recall:")
    print(
        f"Recall@1: "
        f"{results['i2t']['R1']:.2f}"
    )
    print(
        f"Recall@5: "
        f"{results['i2t']['R5']:.2f}"
    )
    print(
        f"Recall@10: "
        f"{results['i2t']['R10']:.2f}"
    )

    print("-" * 60)


# ============================================================
# CCA
# ============================================================

def run_cca(
    train_se1: np.ndarray,
    train_se2: np.ndarray,
    test_se1: np.ndarray,
    test_se2: np.ndarray,
    pair_indices: np.ndarray,
    n_components: int,
):
    """
    Fit CCA using ONLY selected real paired samples.

    Then project the entire fixed SE train/test sets.

    This intentionally follows the original SUE implementation:
        cca.fit(...)
        projection1 = cca.x_rotations_
        projection2 = cca.y_rotations_
        X_projected = X @ projection
    """

    if len(pair_indices) <= n_components:
        raise ValueError(
            f"CCA n_components={n_components}, "
            f"but only {len(pair_indices)} pairs supplied.\n"
            "Use more pairs or fewer CCA components."
        )

    paired1 = train_se1[pair_indices]
    paired2 = train_se2[pair_indices]

    print(
        f"Fitting CCA using "
        f"{len(pair_indices)} real pairs..."
    )

    cca = CCA(
        n_components=n_components
    )

    cca.fit(
        paired1,
        paired2,
    )

    projection1 = cca.x_rotations_
    projection2 = cca.y_rotations_

    # Project entire distributions
    train_cca1 = (
        train_se1
        @ projection1
    )

    train_cca2 = (
        train_se2
        @ projection2
    )

    test_cca1 = (
        test_se1
        @ projection1
    )

    test_cca2 = (
        test_se2
        @ projection2
    )

    return {
        "cca": cca,

        "projection1": projection1,
        "projection2": projection2,

        "train1": train_cca1,
        "train2": train_cca2,

        "test1": test_cca1,
        "test2": test_cca2,
    }


# ============================================================
# MMD
# ============================================================

def run_mmd(
    train_cca1: np.ndarray,
    train_cca2: np.ndarray,
    test_cca1: np.ndarray,
    test_cca2: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int = 100,
    batch_size: int = 32,
    n_scales: int = 3,
):
    """
    Run the same MMD refinement used by SUE.

    Important:
        MMD transforms modality1 only.
        modality2 remains unchanged.

    We reset the same seed before every pair condition so that
    MMD initialization / validation split randomness is controlled
    across different n_pairs.
    """

    set_seed(seed)

    print(
        f"Training MMD refinement "
        f"(epochs={epochs}, "
        f"batch_size={batch_size}, "
        f"n_scales={n_scales})..."
    )

    (
        train_mmd1,
        train_mmd2,
        mmd_model,
    ) = fine_tune_alignment_using_mmd_network(
        X=train_cca1,
        Y=train_cca2,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        n_scales=n_scales,
    )

    # --------------------------------------------------------
    # Test:
    # original SUE only transforms modality1 using MMD
    # --------------------------------------------------------

    mmd_model.eval()

    with torch.no_grad():

        test_tensor1 = torch.from_numpy(
            test_cca1
        ).float().to(device)

        test_mmd1 = (
            mmd_model(test_tensor1)
            .detach()
            .cpu()
            .numpy()
        )

    test_mmd2 = test_cca2.astype(float)

    return {
        "model": mmd_model,

        "train1": train_mmd1,
        "train2": train_mmd2,

        "test1": test_mmd1,
        "test2": test_mmd2,
    }


# ============================================================
# Result utilities
# ============================================================

def build_result_row(
    dataset_name,
    seed,
    n_pairs,
    cca_results,
    final_results,
):
    """
    Convert experiment results to one flat CSV row.
    """

    return {
        "dataset": dataset_name,
        "seed": seed,
        "n_pairs": n_pairs,

        # ---------------- CCA only ----------------

        "cca_t2i_R1":
            cca_results["t2i"]["R1"],

        "cca_t2i_R5":
            cca_results["t2i"]["R5"],

        "cca_t2i_R10":
            cca_results["t2i"]["R10"],

        "cca_i2t_R1":
            cca_results["i2t"]["R1"],

        "cca_i2t_R5":
            cca_results["i2t"]["R5"],

        "cca_i2t_R10":
            cca_results["i2t"]["R10"],

        # ---------------- CCA + MMD ----------------

        "final_t2i_R1":
            final_results["t2i"]["R1"],

        "final_t2i_R5":
            final_results["t2i"]["R5"],

        "final_t2i_R10":
            final_results["t2i"]["R10"],

        "final_i2t_R1":
            final_results["i2t"]["R1"],

        "final_i2t_R5":
            final_results["i2t"]["R5"],

        "final_i2t_R10":
            final_results["i2t"]["R10"],
    }


def save_results_csv(
    rows,
    output_path: Path,
):
    """
    Save all pair-removal conditions to CSV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if len(rows) == 0:
        raise ValueError(
            "No result rows to save."
        )

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Results saved to:\n"
        f"  {output_path}"
    )


def save_diagnostics(
    output_dir: Path,
    dataset_name: str,
    seed: int,
    n_pairs: int,
    pair_indices: np.ndarray,
    projection1: np.ndarray,
    projection2: np.ndarray,
):
    """
    Save selected pairs and CCA projections.

    Useful later for:
        mapping instability
        projection stability
        N_crit analysis
    """

    diagnostic_dir = (
        output_dir
        / dataset_name
        / f"seed{seed}"
        / "diagnostics"
    )

    diagnostic_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic_path = (
        diagnostic_dir
        / f"pairs_{n_pairs}.npz"
    )

    np.savez_compressed(
        diagnostic_path,

        pair_indices=pair_indices,

        projection1=projection1,
        projection2=projection2,
    )


def save_mmd_model(
    model,
    output_dir: Path,
    dataset_name: str,
    seed: int,
    n_pairs: int,
):
    """
    Optional MMD checkpoint.
    """

    model_dir = (
        output_dir
        / dataset_name
        / f"seed{seed}"
        / "mmd_models"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_dir
        / f"mmd_pairs_{n_pairs}.pth"
    )

    torch.save(
        model.state_dict(),
        model_path,
    )


# ============================================================
# Main experiment
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Controlled Pair Removal "
            "experiment for SUE"
        )
    )

    parser.add_argument(
        "data",
        type=str,
        help="Dataset name, e.g. flickr30",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Fixed SE / experiment seed",
    )

    parser.add_argument(
        "--pairs",
        type=int,
        nargs="+",
        default=[
            500,
            300,
            200,
            100,
            50,
            20,
            10,
        ],
        help=(
            "Nested pair counts to evaluate. "
            "Default: 500 300 200 100 50 20 10"
        ),
    )

    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help=(
            "Optional explicit fixed-SE .npz path"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="PyTorch device",
    )

    parser.add_argument(
        "--mmd_epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--mmd_batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--mmd_scales",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--save_mmd_models",
        action="store_true",
        help="Save each trained MMD model",
    )

    args = parser.parse_args()

    # ========================================================
    # Setup
    # ========================================================

    set_seed(args.seed)

    dataset_name = args.data

    config = load_config(
        dataset_name
    )

    n_components = int(
        config["n_components"]
    )

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        print(
            "CUDA not available. "
            "Falling back to CPU."
        )

        device = torch.device("cpu")

    else:
        device = torch.device(
            args.device
        )

    output_dir = Path(
        args.output_dir
    )

    # ========================================================
    # Load fixed SE
    # ========================================================

    cache = load_fixed_se_cache(
        dataset_name=dataset_name,
        seed=args.seed,
        cache_path=args.cache,
    )

    train_se1 = cache["train_se1"]
    train_se2 = cache["train_se2"]

    test_se1 = cache["test_se1"]
    test_se2 = cache["test_se2"]

    pair_order = cache["pair_order"]

    # ========================================================
    # Validate requested pair counts
    # ========================================================

    pair_numbers = args.pairs

    for n_pairs in pair_numbers:

        if n_pairs <= 0:
            raise ValueError(
                "n_pairs must be > 0.\n"
                "Zero-pair requires a separate "
                "replacement method and should not "
                "be implemented using ordinary CCA."
            )

        if n_pairs > len(pair_order):
            raise ValueError(
                f"Requested n_pairs={n_pairs}, "
                f"but master pair pool contains only "
                f"{len(pair_order)} pairs."
            )

        if n_pairs <= n_components:
            raise ValueError(
                f"n_pairs={n_pairs} is too small for "
                f"CCA n_components={n_components}."
            )

    # ========================================================
    # Print experiment setup
    # ========================================================

    print("=" * 70)
    print("SUE Controlled Pair Removal Experiment")
    print("=" * 70)

    print(
        f"Dataset          : {dataset_name}"
    )

    print(
        f"Seed             : {args.seed}"
    )

    print(
        f"Device           : {device}"
    )

    print(
        f"CCA components   : {n_components}"
    )

    print(
        f"Master pair pool : {len(pair_order)}"
    )

    print(
        f"Pair sweep       : {pair_numbers}"
    )

    print(
        f"MMD epochs       : {args.mmd_epochs}"
    )

    print(
        f"MMD batch size   : "
        f"{args.mmd_batch_size}"
    )

    print(
        f"MMD scales       : "
        f"{args.mmd_scales}"
    )

    print("=" * 70)
    print()

    # ========================================================
    # Pair sweep
    # ========================================================

    all_results = []

    for experiment_idx, n_pairs in enumerate(
        pair_numbers,
        start=1,
    ):

        print()
        print("#" * 70)

        print(
            f"Experiment "
            f"{experiment_idx}/{len(pair_numbers)}"
        )

        print(
            f"N_PAIRS = {n_pairs}"
        )

        print("#" * 70)
        print()

        # ----------------------------------------------------
        # Nested pair selection
        #
        # P10 ⊂ P20 ⊂ ... ⊂ P500
        # ----------------------------------------------------

        pair_indices = (
            pair_order[:n_pairs]
            .copy()
        )

        print(
            f"Selected {len(pair_indices)} "
            f"nested real pairs."
        )

        # ----------------------------------------------------
        # Stage 1: CCA
        # ----------------------------------------------------

        cca_output = run_cca(
            train_se1=train_se1,
            train_se2=train_se2,

            test_se1=test_se1,
            test_se2=test_se2,

            pair_indices=pair_indices,

            n_components=n_components,
        )

        cca_recall = (
            compute_bidirectional_recall(
                image_embeddings=
                    cca_output["test1"],

                text_embeddings=
                    cca_output["test2"],
            )
        )

        print()

        print_recall(
            title=(
                f"[{n_pairs} pairs] "
                f"CCA-only"
            ),
            results=cca_recall,
        )

        # ----------------------------------------------------
        # Save CCA diagnostics
        # ----------------------------------------------------

        save_diagnostics(
            output_dir=output_dir,
            dataset_name=dataset_name,
            seed=args.seed,
            n_pairs=n_pairs,

            pair_indices=pair_indices,

            projection1=
                cca_output["projection1"],

            projection2=
                cca_output["projection2"],
        )

        # ----------------------------------------------------
        # Stage 2: MMD
        # ----------------------------------------------------

        print()

        mmd_output = run_mmd(
            train_cca1=
                cca_output["train1"],

            train_cca2=
                cca_output["train2"],

            test_cca1=
                cca_output["test1"],

            test_cca2=
                cca_output["test2"],

            device=device,

            # Same seed for every n_pairs:
            # control MMD randomness
            seed=args.seed,

            epochs=args.mmd_epochs,

            batch_size=
                args.mmd_batch_size,

            n_scales=
                args.mmd_scales,
        )

        final_recall = (
            compute_bidirectional_recall(
                image_embeddings=
                    mmd_output["test1"],

                text_embeddings=
                    mmd_output["test2"],
            )
        )

        print()

        print_recall(
            title=(
                f"[{n_pairs} pairs] "
                f"CCA + MMD"
            ),
            results=final_recall,
        )

        # ----------------------------------------------------
        # Optional MMD model save
        # ----------------------------------------------------

        if args.save_mmd_models:

            save_mmd_model(
                model=
                    mmd_output["model"],

                output_dir=
                    output_dir,

                dataset_name=
                    dataset_name,

                seed=
                    args.seed,

                n_pairs=
                    n_pairs,
            )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        row = build_result_row(
            dataset_name=
                dataset_name,

            seed=
                args.seed,

            n_pairs=
                n_pairs,

            cca_results=
                cca_recall,

            final_results=
                final_recall,
        )

        all_results.append(
            row
        )

        # ----------------------------------------------------
        # Incremental save:
        # prevents losing finished experiments
        # ----------------------------------------------------

        pairs_str = "_".join(str(p) for p in pair_numbers)
        result_path = (
            output_dir
            / dataset_name
            / f"seed{args.seed}"
            / f"results_{pairs_str}.csv"
        )

        save_results_csv(
            all_results,
            result_path,
        )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    header = (
        f"{'Pairs':>7} | "
        f"{'CCA T2I@10':>11} | "
        f"{'CCA I2T@10':>11} | "
        f"{'Final T2I@10':>13} | "
        f"{'Final I2T@10':>13}"
    )

    print(header)
    print("-" * len(header))

    for row in all_results:

        print(
            f"{row['n_pairs']:>7} | "
            f"{row['cca_t2i_R10']:>11.2f} | "
            f"{row['cca_i2t_R10']:>11.2f} | "
            f"{row['final_t2i_R10']:>13.2f} | "
            f"{row['final_i2t_R10']:>13.2f}"
        )

    print("=" * 100)

    print()
    print("Experiment completed.")

    print(
        f"Results directory:\n"
        f"  {output_dir / dataset_name / f'seed{args.seed}_{pairs_str}'}"
    )


if __name__ == "__main__":
    main()

