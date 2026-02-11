from datetime import datetime
from airflow import DAG
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.operators.empty import EmptyOperator

from dag_parts.dwh_incremental_loading_task_flow import get_incr_warehouse_loading_task_flow

with DAG(
    dag_id="warehouse_incremental_loading",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    get_incr_warehouse_loading_task_flow()
