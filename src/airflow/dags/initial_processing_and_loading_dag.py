from datetime import datetime
from airflow import DAG

from dag_parts.processing_task_flow import get_processing_task_flow

with DAG(
    dag_id="initial_processing_and_loading",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    get_processing_task_flow()
