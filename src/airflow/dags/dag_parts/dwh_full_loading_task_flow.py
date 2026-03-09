import os
from airflow.models.baseoperator import BaseOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.operators.empty import EmptyOperator

from dag_parts.common import (
    PATH_TO_DB_TABLES_CREATION_SCRIPTS,
    PATH_TO_DB_TABLES_FULL_LOADING_SCRIPTS
)


def read_query_from_sql_file(path: str) -> str:
    assert path.strip().endswith('.sql')

    with open(path, 'r') as f:
        query = f.read()

    return query


def get_full_warehouse_loading_task_flow() -> BaseOperator:
    start_task = EmptyOperator(task_id="start")

    raw_tables = ["raw_fight_stats", "raw_fighters_current_stats"]
    dim_tables = ["dim_date", "dim_fight_time_format", "dim_fighter", "dim_gender",
                  "dim_method", "dim_result", "dim_weight_class"]
    fact_tables = ["fact_fight"]

    dummy_raw_loading_task = EmptyOperator(task_id="finalize_raw_loading")
    dummy_dim_loading_task = EmptyOperator(task_id="finalize_dim_loading")

    raw_loading_tasks = [
        start_task >>
        SQLExecuteQueryOperator(
            task_id=f"create_{raw_table}",
            conn_id="warehouse_db",
            sql=read_query_from_sql_file(
                os.path.join(PATH_TO_DB_TABLES_CREATION_SCRIPTS,
                             "raw", f"{raw_table}.sql")
            )
        ) >>
        SQLExecuteQueryOperator(
            task_id=f"load_{raw_table}",
            conn_id="warehouse_db",
            sql=read_query_from_sql_file(
                os.path.join(PATH_TO_DB_TABLES_FULL_LOADING_SCRIPTS,
                             "raw", f"{raw_table}.sql")
            )
        ) >>
        dummy_raw_loading_task for raw_table in raw_tables
    ]

    dim_loading_tasks = [
        dummy_raw_loading_task >>
        SQLExecuteQueryOperator(
            task_id=f"create_{dim_table}",
            conn_id="warehouse_db",
            sql=read_query_from_sql_file(
                os.path.join(PATH_TO_DB_TABLES_CREATION_SCRIPTS,
                             "dimensions", f"{dim_table}.sql")
            )
        ) >>
        SQLExecuteQueryOperator(
            task_id=f"load_{dim_table}",
            conn_id="warehouse_db",
            sql=read_query_from_sql_file(
                os.path.join(PATH_TO_DB_TABLES_FULL_LOADING_SCRIPTS,
                             "dimensions", f"{dim_table}.sql")
            )
        ) >>
        dummy_dim_loading_task for dim_table in dim_tables
    ]

    fact_loading_tasks = [
        dummy_dim_loading_task >>
        SQLExecuteQueryOperator(
            task_id=f"create_{fact_table}",
            conn_id="warehouse_db",
            sql=read_query_from_sql_file(
                os.path.join(PATH_TO_DB_TABLES_CREATION_SCRIPTS,
                             "facts", f"{fact_table}.sql")
            )
        ) >>
        SQLExecuteQueryOperator(
            task_id=f"load_{fact_table}",
            conn_id="warehouse_db",
            sql=read_query_from_sql_file(
                os.path.join(PATH_TO_DB_TABLES_FULL_LOADING_SCRIPTS,
                             "facts", f"{fact_table}.sql")
            )
        ) for fact_table in fact_tables
    ]

    end_task = EmptyOperator(task_id="end")

    fact_loading_tasks >> end_task

    return start_task
