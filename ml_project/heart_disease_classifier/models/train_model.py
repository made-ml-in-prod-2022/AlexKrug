import json
import logging
import sys

import click
import pandas as pd

from heart_disease_classifier.data import read_data, split_train_val_data
from heart_disease_classifier.entities.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params
from heart_disease_classifier.features import make_features
from heart_disease_classifier.features.build_features import extract_target, build_transformer
from heart_disease_classifier.models import (
    train_model_func,
    serialize_model,
    predict_model_func,
    evaluate_model,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
file_log = logging.FileHandler("data/interim/process_train.log")
logger.setLevel(logging.INFO)
handler.setLevel(logging.DEBUG)
file_log.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(file_log)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info("Start train pipeline...")
    logger.info(f"Params: {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape: {data.shape}")
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape: {train_df.shape}")
    logger.info(f"val_df.shape: {val_df.shape}")

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    train_features = make_features(transformer, train_df)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    logger.info(f"train_features.shape: {train_features.shape}")
    logger.info(f"{training_pipeline_params.train_params.model_type} training...")

    model = train_model_func(
        train_features, train_target, training_pipeline_params.train_params
    )

    val_features = make_features(transformer, val_df)
    val_target = extract_target(val_df, training_pipeline_params.feature_params)

    logger.info(f"val_features.shape: {val_features.shape}")
    predicts = predict_model_func(
        model,
        val_features
    )

    metrics = evaluate_model(
        predicts,
        val_target
    )
    logger.info(f"Writing metrics in: {training_pipeline_params.metric_path} ...")
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics: {metrics}")

    logger.info(f"Serialization in: {training_pipeline_params.output_model_path}")
    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)
    logger.info(f"Serialization completed: {path_to_model}")
    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()