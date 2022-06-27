import os
import pickle

import click
import pandas as pd


PROCESSED_DATA_PATH = "data_norm.csv"
MODEL_NAME = "model.pkl"
PRED_PATH = "predictions.csv"


@click.command()
@click.option("--input_dir")
@click.option("--model_dir")
@click.option("--pred_dir")
def predict(input_dir: str, model_dir: str, pred_dir: str) -> None:
    path = os.path.join(input_dir, PROCESSED_DATA_PATH)
    data = pd.read_csv(path)
    os.makedirs(pred_dir, exist_ok=True)

    model_path = os.path.join(model_dir, MODEL_NAME)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    pred = pd.DataFrame(model.predict(data))
    pred_path = os.path.join(pred_dir, PRED_PATH)
    pred.to_csv(pred_path, index=False)


if __name__ == '__main__':
    predict()
