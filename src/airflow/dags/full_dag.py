from datetime import datetime, timezone
import os
import secrets
from airflow import DAG
from airflow.sdk import Param
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

from dag_parts.crawling_task_flow import get_crawling_task_flow, check_if_incremental_or_full_load_run
from dag_parts.processing_task_flow import get_processing_task_flow
from dag_parts.dwh_full_loading_task_flow import get_full_warehouse_loading_task_flow
from dag_parts.dwh_incremental_loading_task_flow import get_incr_warehouse_loading_task_flow
from dag_parts.serving_views_creation import get_serving_views_creation_task_flow


def determine_version_id(param_version_id: str | None) -> str:
    default_version_id = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H_%M_%S_%f')}_{secrets.token_hex(3)}"

    print(f"Resulting Version ID: {param_version_id or default_version_id}")
    return param_version_id or default_version_id


with DAG(
    dag_id="full_dag",
    params={
        "is_incremental": Param(True, type="boolean"),
        "lookup_days": Param(15, type="integer"),
        "version_id": Param("", type=["null", "string"])
    },
    start_date=datetime(2025, 1, 1),
    schedule="30 9 * * 1",
    catchup=False,
) as dag:
    determine_version_id_task = PythonOperator(
        task_id="determine_version_id",
        python_callable=determine_version_id,
        op_kwargs={
            "param_version_id": "{{ params.version_id }}"
        }
    )
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

    dwh_join = EmptyOperator(
        task_id="dwh_loading_join",
        trigger_rule="none_failed_min_one_success",
    )

    with TaskGroup(group_id="serving_views_creation") as serving_creation_group:
        get_serving_views_creation_task_flow()

    determine_version_id_task >> crawl_data_group
    crawl_data_group >> crawl_join >> proc_and_load_group >> wh_circuit_operator
    wh_circuit_operator >> [full_dwh_load_group, incr_dwh_load_group] >> dwh_join
    dwh_join >> serving_creation_group
