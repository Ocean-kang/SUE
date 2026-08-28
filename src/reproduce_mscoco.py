import json

from data import (
    load_dataset,
    create_weakly_parallel_data,
)
from trainer import Trainer


def main():

    with open(
        "../configs/mscoco_config.json"
    ) as f:
        configs = json.load(f)

    train_set, test_set = load_dataset(
        "mscoco",
        n_test=configs["n_test"]
    )

    train_set = create_weakly_parallel_data(
        train_set,
        n_parallel=configs["n_parallel"]
    )

    trainer = Trainer(
        dataset_name="mscoco",
        n_parallel=configs["n_parallel"],
        n_components=configs["n_components"],
        configs=configs,
    )

    trainer.fit(
        train_set=train_set,
        with_se=True,
        with_cca=True,
        with_mmd=True,
    )

    trainer.test(
        test_set=test_set
    )


if __name__ == "__main__":
    main()