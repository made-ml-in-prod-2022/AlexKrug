from datetime import timedelta

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


GENERATED_DATA_PATH = "/data/raw/{{ ds }}"
PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
MODEL_PATH = "/data/models/svm/{{ ds }}"
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
        "train",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(0),
) as dag:
    data_sensor = FileSensor(
        task_id="data_sensor",
        filepath=f"{GENERATED_DATA_PATH}/data.csv",
        poke_interval=10,
        retries=10,
        mode="reschedule",
    )

    target_sensor = FileSensor(
        task_id="target_sensor",
        filepath=f"{GENERATED_DATA_PATH}/target.csv",
        poke_interval=10,
        retries=10,
        mode="reschedule",
    )

    prepare_data = DockerOperator(
        image="airflow-preparing",
        command=f"--input_dir {GENERATED_DATA_PATH} --output_dir {PROCESSED_DATA_PATH}",
        task_id="docker-airflow-preparing",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT_SOURCE]
    )

    split_data = DockerOperator(
        image="airflow-splitting",
        command=f"--input_dir {PROCESSED_DATA_PATH}",
        task_id="docker-airflow-splitting",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT_SOURCE]
    )

    train_model = DockerOperator(
        image="airflow-training",
        command=f"--input_dir {PROCESSED_DATA_PATH} --output_dir {MODEL_PATH}",
        task_id="docker-airflow-training",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT_SOURCE]
    )

    validate_model = DockerOperator(
        image="airflow-validation",
        command=f"--model_dir {MODEL_PATH} --data_dir {PROCESSED_DATA_PATH}",
        task_id="docker-airflow-validation",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[MOUNT_SOURCE]
    )

    [data_sensor, target_sensor] >> prepare_data >> split_data >> train_model >> validate_model
