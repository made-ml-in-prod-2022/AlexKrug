import json
import logging
import sys

import click
import pandas as pd

from heart_disease_classifier.data.make_dataset import write_data, read_data
from heart_disease_classifier.entities.predict_pipeline_params import PredictPipelineParams, read_predict_pipeline_params
from heart_disease_classifier.features import make_features
from heart_disease_classifier.features.build_features import build_transformer
from heart_disease_classifier.models import deserialize_model, predict_model_func

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
file_log = logging.FileHandler("data/interim/predict.log")
logger.setLevel(logging.INFO)
handler.setLevel(logging.DEBUG)
file_log.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(file_log)


def predict_pipeline(predict_pipeline_params: PredictPipelineParams):
    logger.info(f"start predict pipeline with params {predict_pipeline_params}")
    data = read_data(predict_pipeline_params.input_data_path)
    logger.info(f"data.shape: {data.shape}")

    transformer = build_transformer(predict_pipeline_params.feature_params)
    transformer.fit(data)
    predict_features = make_features(transformer, data)

    logger.info(f"predict_features.shape: {predict_features.shape}")

    model = deserialize_model(predict_pipeline_params.output_model_path)
    logger.info(f"Uploaded model: {model}")

    predicts = predict_model_func(
        model,
        predict_features
    )

    logger.info(f"Predicted values shape: {predicts.shape}")
    data["predicts"] = predicts
    
    logger.info(f"Writing predicts ...: {predict_pipeline_params.output_data_path}")
    result_path = write_data(data, predict_pipeline_params.output_data_path)

    return result_path


@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    params = read_predict_pipeline_params(config_path)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_command()