import os
import json
import random
import argparse

import numpy as np
import torch

from data import load_dataset, create_weakly_parallel_data
from trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(dataset):
    with open(f"../configs/{dataset}_config.json", "r") as f:
        return json.load(f)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=str)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    configs = load_config(args.data)

    # -----------------------------------
    # 1. 原始 train / test 固定
    # -----------------------------------

    train_set, test_set = load_dataset(
        args.data,
        n_test=configs["n_test"]
    )

    # -----------------------------------
    # 2. 永远只建立一次 500-pair master split
    # -----------------------------------

    n_master_pairs = 500

    master_train_set = create_weakly_parallel_data(
        train_set,
        n_parallel=n_master_pairs
    )

    # -----------------------------------
    # 3. 只训练 SE
    # -----------------------------------

    trainer = Trainer(
        dataset_name=args.data,
        n_parallel=n_master_pairs,
        n_components=configs["n_components"],
        configs=configs
    )

    trainer.fit(
        train_set=master_train_set,
        with_se=True,
        with_cca=False,
        with_mmd=False
    )

    train_se1 = trainer.embeddings1.copy()
    train_se2 = trainer.embeddings2.copy()

    # -----------------------------------
    # 4. test 同样经过固定 SE
    # -----------------------------------

    trainer.spectralnet1.transform(
        test_set[0].float()
    )

    test_se1 = trainer.spectralnet1.embeddings_.copy()

    trainer.spectralnet2.transform(
        test_set[1].float()
    )

    test_se2 = trainer.spectralnet2.embeddings_.copy()

    # -----------------------------------
    # 5. 最后 500 个就是 master real pairs
    # -----------------------------------

    n_train = len(train_se1)

    pair_pool = np.arange(
        n_train - n_master_pairs,
        n_train
    )

    # 固定 pair 顺序
    rng = np.random.default_rng(args.seed)

    pair_order = rng.permutation(pair_pool)

    # -----------------------------------
    # 6. 保存
    # -----------------------------------

    os.makedirs("../artifacts/fixed_se", exist_ok=True)

    save_path = (
        f"../artifacts/fixed_se/"
        f"{args.data}_seed{args.seed}.npz"
    )

    np.savez_compressed(
        save_path,

        train_se1=train_se1,
        train_se2=train_se2,

        test_se1=test_se1,
        test_se2=test_se2,

        pair_order=pair_order
    )

    print("Saved:", save_path)

    print("train_se1:", train_se1.shape)
    print("train_se2:", train_se2.shape)

    print("test_se1:", test_se1.shape)
    print("test_se2:", test_se2.shape)

    print("pair pool:", len(pair_order))


if __name__ == "__main__":
    main()