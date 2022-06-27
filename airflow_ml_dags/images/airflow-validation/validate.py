import os
import pickle

import click
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

VAL_DATA_PATH = "val_data.csv"
VAL_TARGET_PATH = "val_target.csv"
MODEL_PATH = 'model.pkl'
METRICS_PATH = 'metrics.txt'


@click.command()
@click.option("--model_dir")
@click.option("--data_dir")
def validate(model_dir: str, data_dir: str) -> None:
    data = pd.read_csv(os.path.join(data_dir, VAL_DATA_PATH))
    target = pd.read_csv(os.path.join(data_dir, VAL_TARGET_PATH))
    with open(os.path.join(model_dir, MODEL_PATH), 'rb') as fin:
        model = pickle.load(fin)

    target = target["target"]
    predictions = model.predict(data)
    f1 = f1_score(target, predictions)
    roc_auc = roc_auc_score(target, predictions)
    with open(os.path.join(model_dir, METRICS_PATH), "w") as f:
        f.write(f"f1_score: {f1}, roc_auc: {roc_auc}")


if __name__ == '__main__':
    validate()
