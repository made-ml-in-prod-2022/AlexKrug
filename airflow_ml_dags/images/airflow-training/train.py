import os
import pickle

import click
import pandas as pd
from sklearn.svm import SVC


TRAIN_DATA_PATH = "train_data.csv"
TRAIN_TARGET_PATH = "train_target.csv"
MODEL_NAME = 'model.pkl'


@click.command()
@click.option("--input_dir")
@click.option("--output_dir")
def train(input_dir: str, output_dir: str) -> None:
    data = pd.read_csv(os.path.join(input_dir, TRAIN_DATA_PATH))
    target = pd.read_csv(os.path.join(input_dir, TRAIN_TARGET_PATH))
    model = SVC()
    model.fit(data, target["target"])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, MODEL_NAME), 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train()
