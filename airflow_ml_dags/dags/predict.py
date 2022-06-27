from datetime import timedelta

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models import Variable
from docker.types import Mount

from vars import MODEL_PATH

PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
PREDICTIONS_PATH = "/data/predictions/{{ ds }}"
MOUNT_SOURCE = Mount(
    source="C:/Users/Sasha/OneDrive/Рабочий стол/MADE/2 сем/ML in prod/AlexKrug/airflow_ml_dags/data/",
    target="/data",
    type='bind'
    )


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(0),
) as dag:
    data_sensor = FileSensor(
        task_id="data_sensor",
        filepath=f"{PROCESSED_DATA_PATH}/data_norm.csv",
        poke_interval=10,
        retries=10,
        mode="reschedule",
    )

    model_sensor = FileSensor(
        task_id="model_sensor",
        filepath=f"{MODEL_PATH}/model.pkl",
        poke_interval=10,
        retries=10,
        mode="reschedule",
    )

    predict = DockerOperator(
        image="airflow-prediction",
        command=f"--input_dir {PROCESSED_DATA_PATH} --model_dir {MODEL_PATH} --pred_dir {PREDICTIONS_PATH}",
        task_id="docker-airflow-prediction",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT_SOURCE]
    )

    [data_sensor, model_sensor] >> predict
