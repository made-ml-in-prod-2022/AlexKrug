import os
from typing import List

from py._path.local import LocalPath

from heart_disease_classifier.models.train_model import train_pipeline
from heart_disease_classifier.entities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
)


def test_train_e2e(
    tmpdir: str,
    dataset_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
    features_to_drop: List[str],
):
    expected_output_model_path = tmpdir + "/model.pkl"
    expected_metric_path = tmpdir + "/metrics.json"
    print("here we go again", expected_metric_path)
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=42),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop,
        ),
        train_params=TrainingParams(model_type="CatBoostClassifier"),
    )
    real_model_path, metrics = train_pipeline(params)
    assert metrics["roc_auc"] > 0.5
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)