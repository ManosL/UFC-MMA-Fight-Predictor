import os
import sys
from datetime import datetime
from airflow import DAG
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.operators.empty import EmptyOperator

PATH_TO_DB_TABLES_CREATION_SCRIPTS = "/opt/airflow/sql/creation"
PATH_TO_DB_TABLES_LOADING_SCRIPTS  = "/opt/airflow/sql/full_loading"


def read_query_from_sql_file(path):
    assert path.strip().endswith('.sql')

    with open(path, 'r') as f:
        query = f.read()

    return query


with DAG(
    dag_id="warehouse_full_loading",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    start_task = EmptyOperator(task_id="start")

    dim_tables = ["dim_date", "dim_fight_time_format", "dim_fighter",
                  "dim_gender", "dim_method", "dim_result", "dim_weight_class"]
    fact_tables = ["fact_fight"]

    dim_loading_tasks  = [
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
                os.path.join(PATH_TO_DB_TABLES_LOADING_SCRIPTS,
                             "dimensions", f"{dim_table}.sql")
            )
        ) for dim_table in dim_tables
    ]

    dummy_dim_loading_task = EmptyOperator(task_id="finalize_dim_loading")

    fact_loading_tasks = [
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
                os.path.join(PATH_TO_DB_TABLES_LOADING_SCRIPTS,
                             "facts", f"{fact_table}.sql")
            )
        ) for fact_table in fact_tables
    ]

    end_task = EmptyOperator(task_id="end")

    start_task >> dim_loading_tasks >> dummy_dim_loading_task
    dummy_dim_loading_task >> fact_loading_tasks >> end_task