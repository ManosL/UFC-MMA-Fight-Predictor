from datetime import datetime
from airflow import DAG

from dag_parts.serving_views_creation import get_serving_views_creation_task_flow

with DAG(
    dag_id="serving_views_creation",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    get_serving_views_creation_task_flow()
