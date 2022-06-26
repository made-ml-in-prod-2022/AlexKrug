import logging

import numpy as np
import pandas as pd
import requests
import click


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.command()
@click.option(
    "--host",
    default="127.0.0.1")
@click.option(
    "--port",
    default=8000)
@click.option(
    "--datapath",
    default="../ml_project/data/raw/heart_disease.csv")
def make_request_func(host, port, datapath):
    data = pd.read_csv(datapath)
    data.drop("condition", inplace=True, axis=1)
    logger.info(f"Prepare data for requests")
    request_features = list(data.columns)
    for i in range(len(data)):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        logger.info(f"Process data: {request_data}")
        response = requests.get(
            f"http://{host}:{port}/predict/",
            json={"data": [request_data], "features": request_features},
        )
        logger.info(f"Status of the request: {response.status_code}")
        logger.info(f"JSON data of response: {response.json()}")


if __name__ == "__main__":
    make_request_func()
