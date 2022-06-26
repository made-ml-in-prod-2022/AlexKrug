import os
import pytest

from fastapi.testclient import TestClient

from app import app


@pytest.fixture()
def datapath() -> str:
    return "../ml_project/data/raw/heart_disease.csv"


@pytest.fixture()
def model_path() -> str:
    os.environ['PATH_TO_MODEL'] = 'model.pkl'
    return os.getenv("PATH_TO_MODEL")


@pytest.fixture()
def client():
    with TestClient(app) as client:
        yield client


def test_main(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == True


def test_app(client) -> None:
    response = client.get("/wrong_entrypoint")
    assert response.status_code >= 400
