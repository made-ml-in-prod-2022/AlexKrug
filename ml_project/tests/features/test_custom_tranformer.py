from typing import List

import numpy as np
import pandas as pd
import pytest

from heart_disease_classifier.entities.feature_params import FeatureParams
from heart_disease_classifier.data.make_dataset import read_data
from heart_disease_classifier.features.build_features import CustomTransformer, make_features


@pytest.fixture
def feature_params(
    categorical_features: List[str],
    features_to_drop: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
    )
    return params


def test_make_features(
    feature_params: FeatureParams, dataset_path: str,
):
    data = read_data(dataset_path)
    test_data = data.copy()
    test_data[feature_params.numerical_features] = test_data[feature_params.numerical_features] ** 2

    transformer = CustomTransformer(feature_params)
    features = make_features(transformer, data)
    assert not pd.isnull(features).any().any()
    assert features.equals(test_data)