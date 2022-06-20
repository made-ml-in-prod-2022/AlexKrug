from dataclasses import dataclass
from .feature_params import FeatureParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictPipelineParams:
    input_data_path: str
    output_data_path: str
    output_model_path: str
    feature_params: FeatureParams


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(path: str) -> PredictPipelineParams:
    with open(path, "r") as input_stream:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))