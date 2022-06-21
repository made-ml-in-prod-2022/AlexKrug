import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

from heart_disease_classifier.entities.train_params import TrainingParams
from heart_disease_classifier.entities.feature_params import FeatureParams

ClassifierModel = Union[RandomForestClassifier, CatBoostClassifier]


def train_model_func(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> ClassifierModel:
    if train_params.model_type == "CatBoostClassifier":
        model = CatBoostClassifier(
            random_seed=train_params.random_state,
            verbose = 0
        )
    elif train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            random_state=train_params.random_state
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model_func(
    model: ClassifierModel, features: pd.DataFrame
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(target, predicts),
        "f1": f1_score(target, predicts),
        "roc_auc": roc_auc_score(target, predicts),
    }


def serialize_model(model: ClassifierModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def deserialize_model(input_path: str) -> ClassifierModel:
    with open(input_path, "rb") as f:
        result = pickle.load(f)
    return result