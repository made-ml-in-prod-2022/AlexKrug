import os

import click
import pandas as pd
from sklearn.datasets import make_classification


DATA_PATH = "data.csv"
TARGET_PATH = "target.csv"


@click.command("generate")
@click.option("--output_dir")
def get_data_sample(output_dir: str) -> None:
    data, target = make_classification(
        n_classes=2,
        n_features=5,
    )
    data = pd.DataFrame(data)
    target = pd.DataFrame(target, columns=["target"])
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, DATA_PATH), index=False)
    target.to_csv(os.path.join(output_dir, TARGET_PATH), index=False)


if __name__ == '__main__':
    get_data_sample()
