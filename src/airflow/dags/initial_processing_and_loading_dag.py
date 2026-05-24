from datetime import datetime
from airflow import DAG
from airflow.sdk import Param

from dag_parts.processing_task_flow import get_processing_task_flow

with DAG(
    dag_id="initial_processing_and_loading",
    params={
        "version_id": Param("", type=["string"])
    },
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    get_processing_task_flow()
