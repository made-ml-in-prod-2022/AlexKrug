import logging
import os
from typing import List, Union, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist

from heart_disease_classifier.models.model_fit_predict import (
    ClassifierModel,
    predict_model_func,
    deserialize_model
)


HELLO_MESSAGE = "REST service for heart disease classification"
MODEL_PATH = "./model.pkl"


logger = logging.getLogger(__name__)
model: Optional[ClassifierModel] = None
app = FastAPI()


class ClassifierModelInput(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=2, max_items=100)]
    features: List[str]


class ClassifierModelResponse(BaseModel):
    id: str
    prediction: float


def make_predict(
    data: List, features: List[str], model: ClassifierModel,
) -> List[ClassifierModelResponse]:
    try:
        data = pd.DataFrame(data, columns=features)
        ids = [int(x) for x in range(len(data))]
        predicts = predict_model_func(model, data)

        return [
            ClassifierModelResponse(id=id_, prediction=float(pred))
            for id_, pred in zip(ids, predicts)
        ]
    except Exception as ex:
        raise HTTPException(status_code=400)


@app.get("/")
def main():
    return HELLO_MESSAGE


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL", MODEL_PATH)
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)
    model = deserialize_model(model_path)


@app.get("/health")
def health() -> bool:
    return not (model is None)


@app.get("/predict/", response_model=List[ClassifierModelResponse])
def predict(request: ClassifierModelInput):
    return make_predict(request.data, request.features, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
