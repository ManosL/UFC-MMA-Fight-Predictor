import os
from airflow.utils.task_group import TaskGroup
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.operators.empty import EmptyOperator

from dag_parts.common import (
    PATH_TO_DB_TABLES_CREATION_SCRIPTS
)


def read_query_from_sql_file(path):
    assert path.strip().endswith('.sql')

    with open(path, 'r') as f:
        query = f.read()

    return query


def get_serving_views_creation_task_flow():
    start_task = EmptyOperator(task_id="start")

    with TaskGroup(group_id="machine_learning") as machine_learning_group:
        ml_views_creation_db_scripts_path = os.path.join(
            PATH_TO_DB_TABLES_CREATION_SCRIPTS,
            "serving",
            "machine_learning"
        )

        ml_view_names = ["ml_fighters", "ml_fights"]

        ml_view_creation_tasks = [
            SQLExecuteQueryOperator(
                task_id=f"create_view_{ml_view}",
                conn_id="warehouse_db",
                sql=read_query_from_sql_file(
                    os.path.join(ml_views_creation_db_scripts_path,
                                f"{ml_view}.sql")
                )
            ) for ml_view in ml_view_names
        ]

    end_task = EmptyOperator(task_id=f"end")

    start_task >> machine_learning_group >> end_task

    return start_task
