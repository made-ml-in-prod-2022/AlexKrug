from datetime import timedelta

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


DATA_PATH = "/data/raw/{{ ds }}"
default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
        "data_generator",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(0),
) as dag:
    generate = DockerOperator(
        image="airflow-generation",
        command=f"--output_dir {DATA_PATH}",
        task_id="docker-airflow-generation",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="C:/Users/Sasha/OneDrive/Рабочий стол/MADE/2 сем/ML in prod/AlexKrug/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    generate
