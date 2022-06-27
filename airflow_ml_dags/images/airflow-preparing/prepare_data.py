import os

import click
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data.csv"
TARGET_PATH = "target.csv"
NORM_DATA_PATH = "data_norm.csv"


@click.command("prepare")
@click.option("--input_dir")
@click.option("--output_dir")
def prepare_data(input_dir: str, output_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, DATA_PATH))
    target = pd.read_csv(os.path.join(input_dir, TARGET_PATH))
    data_norm = StandardScaler().fit_transform(data)
    data_norm = pd.DataFrame(data=data_norm, columns=data.columns)
    os.makedirs(output_dir, exist_ok=True)
    data_norm.to_csv(os.path.join(output_dir, NORM_DATA_PATH), index=False)
    target.to_csv(os.path.join(output_dir, TARGET_PATH), index=False)


if __name__ == '__main__':
    prepare_data()
