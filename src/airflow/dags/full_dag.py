from datetime import datetime, timezone
import os
import sys
from airflow import DAG
from airflow.sdk import Param, get_current_context
from airflow.models.xcom_arg import XComArg
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonVirtualenvOperator
from airflow.operators.python import BranchPythonOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.operators.empty import EmptyOperator

from dag_parts.crawling_task_flow import get_crawling_task_flow, check_if_incremental_or_full_load_run
from dag_parts.processing_task_flow import get_processing_task_flow
from dag_parts.dwh_full_loading_task_flow import get_full_warehouse_loading_task_flow
from dag_parts.dwh_incremental_loading_task_flow import get_incr_warehouse_loading_task_flow


with DAG(
    dag_id="full_dag",
    params={
        "is_incremental": Param(False, type="boolean"),
        "lookup_days": Param(15, type="integer")
    },
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    with TaskGroup(group_id="crawl_data") as crawl_data_group:
        get_crawling_task_flow()

    crawl_join = EmptyOperator(
        task_id="crawl_join",
        trigger_rule="none_failed_min_one_success",
    )

    with TaskGroup(group_id="initial_processing_and_loading") as proc_and_load_group:
        get_processing_task_flow()

    wh_circuit_operator = \
        BranchPythonOperator(
            task_id="check_if_incremental_wh_load",
            python_callable=check_if_incremental_or_full_load_run,
            op_kwargs={
                "is_incremental": "{{ params.is_incremental }}",
                "incr_task_id": "incr_dwh_load",
                "full_task_id": "full_dwh_load"
            }
        )

    with TaskGroup(group_id="full_dwh_load") as full_dwh_load_group:
        get_full_warehouse_loading_task_flow()

    with TaskGroup(group_id="incr_dwh_load") as incr_dwh_load_group:
        get_incr_warehouse_loading_task_flow()

    crawl_data_group >> crawl_join >> proc_and_load_group >> wh_circuit_operator
    wh_circuit_operator >> [full_dwh_load_group, incr_dwh_load_group]
