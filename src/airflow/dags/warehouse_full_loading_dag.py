from datetime import datetime
from airflow import DAG

from dag_parts.dwh_full_loading_task_flow import get_full_warehouse_loading_task_flow

with DAG(
    dag_id="warehouse_full_loading",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    get_full_warehouse_loading_task_flow()
