from collections.abc import Sequence
from datetime import datetime, timezone
import os
import secrets
from airflow import DAG
from airflow.sdk import Param
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonVirtualenvOperator, PythonOperator
from airflow.operators.empty import EmptyOperator


def determine_version_id(param_version_id: str | None) -> str:
    default_version_id = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H_%M_%S_%f')}_{secrets.token_hex(3)}"

    print(f"Resulting Version ID: {param_version_id or default_version_id}")
    return param_version_id or default_version_id


def get_machine_learning_requirements() -> Sequence[str]:
    with open("/opt/airflow/machine_learning/requirements.txt") as f:
        requirements = [l.strip() for l in f.readlines()
                        if not l.strip().startswith("#")]

    return requirements


def write_and_split_training_data_to_minio_venv(ml_pipeline_run_id: str, folds: int) -> None:
    from write_and_split_training_data_to_minio import main

    main(ml_pipeline_run_id, int(folds))
    return


with DAG(
    dag_id="ml_pipeline",
    params={
        "ml_pipeline_run_id": Param("", type=["null", "string"]),
        "folds": Param(5, type="integer"),
    },
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    ml_requirements = get_machine_learning_requirements()

    start_task = EmptyOperator(task_id="start_ml_pipeline")

    determine_version_id_task = PythonOperator(
        task_id="determine_version_id",
        python_callable=determine_version_id,
        op_kwargs={
            "param_version_id": "{{ params.ml_pipeline_run_id }}"
        }
    )

    write_and_split_training_data_to_minio_task = PythonVirtualenvOperator(
        task_id="write_and_split_training_data_to_minio",
        python_callable=write_and_split_training_data_to_minio_venv,
        op_kwargs={
            "ml_pipeline_run_id": "{{ ti.xcom_pull(task_ids='determine_version_id') or params.ml_pipeline_run_id }}",
            "folds": "{{ params.folds }}"
        },
        requirements=ml_requirements,
        system_site_packages=False,
    )

    end_task = EmptyOperator(task_id="end_processing")

    start_task >> determine_version_id_task >> write_and_split_training_data_to_minio_task >> end_task
