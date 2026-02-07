from datetime import datetime
import os
import sys
from airflow import DAG
from airflow.sdk import Param, get_current_context
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonVirtualenvOperator
from airflow.operators.python import BranchPythonOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.operators.empty import EmptyOperator

PATH_TO_DB_TABLES_CREATION_SCRIPTS = "/opt/airflow/sql/creation"
PATH_TO_DB_TABLES_FULL_LOADING_SCRIPTS  = "/opt/airflow/sql/full_loading"
PATH_TO_DB_TABLES_INCR_LOADING_SCRIPTS  = "/opt/airflow/sql/incremental_loading"


def read_query_from_sql_file(path):
    assert path.strip().endswith('.sql')

    with open(path, 'r') as f:
        query = f.read()

    return query

def crawl_from_scrapyd_venv(
    base_url,
    project_name,
    spider_name,
    is_incremental="False",
    lookup_days=15,
):
    sys.path.append('/opt/airflow/dags')
    from scripts.crawl_from_scrapyd import crawl_from_scrapyd

    crawl_from_scrapyd(
        base_url,
        project_name,
        spider_name,
        is_incremental,
        int(lookup_days)
    )


def check_if_incremental_or_full_load_run(is_incremental, incr_task_id, full_task_id):
    print(is_incremental)
    return incr_task_id if is_incremental.lower() == "true" else full_task_id


def get_full_crawl_task_flow():
    start_task = EmptyOperator(task_id="full_crawl_start")

    crawl_fights_task = PythonVirtualenvOperator(
        task_id="full_crawl_fights",
        python_callable=crawl_from_scrapyd_venv,
        op_kwargs={
            "base_url": "http://scrapy-server:6800",
            "project_name": "UFCStats_Crawlers",
            "spider_name": "event_spider",
            "is_incremental": "False",
            "lookup_days": 15
        },
        requirements=[
            "requests",
        ],
        system_site_packages=False,  # important: isolate from Airflow deps
    )

    crawl_fighters_task = PythonVirtualenvOperator(
        task_id="full_crawl_fighters",
        python_callable=crawl_from_scrapyd_venv,
        op_kwargs={
            "base_url": "http://scrapy-server:6800",
            "project_name": "UFCStats_Crawlers",
            "spider_name": "fighters_spider",
            "is_incremental": "False",
            "lookup_days": 15
        },
        requirements=[
            "requests",
        ],
        system_site_packages=False,  # important: isolate from Airflow deps
    )

    end_task = EmptyOperator(task_id="full_crawl_end")

    start_task >> [crawl_fights_task, crawl_fighters_task] >> end_task
    return start_task


def get_incremental_crawl_task_flow():
    start_task = EmptyOperator(task_id="incremental_crawl_start")

    crawl_fights_task = PythonVirtualenvOperator(
        task_id="incremental_crawl_fights",
        python_callable=crawl_from_scrapyd_venv,
        op_kwargs={
            "base_url": "http://scrapy-server:6800",
            "project_name": "UFCStats_Crawlers",
            "spider_name": "event_spider",
            "is_incremental": "True",
            "lookup_days": "{{ params.lookup_days }}"
        },
        requirements=[
            "requests",
        ],
        system_site_packages=False,  # important: isolate from Airflow deps
    )

    crawl_fighters_task = PythonVirtualenvOperator(
        task_id="incremental_crawl_fighters",
        python_callable=crawl_from_scrapyd_venv,
        op_kwargs={
            "base_url": "http://scrapy-server:6800",
            "project_name": "UFCStats_Crawlers",
            "spider_name": "fighters_spider",
            "is_incremental": "True",
            "lookup_days": "{{ params.lookup_days }}"
        },
        requirements=[
            "requests",
        ],
        system_site_packages=False,  # important: isolate from Airflow deps
    )

    end_task = EmptyOperator(task_id="incremental_crawl_end")

    start_task >> crawl_fights_task >> crawl_fighters_task >> end_task
    return start_task


def get_processing_requirements():
    with open("/opt/airflow/processing/requirements.txt") as f:
        requirements = [l.strip() for l in f.readlines()
                        if not l.strip().startswith("#")]

    return requirements


def clean_and_process_fights_venv():
    sys.path.append("/opt/airflow/processing")
    from clean_and_process_fights import main

    main()
    return


def configure_fighters_dataset_venv():
    sys.path.append("/opt/airflow/processing")
    from Configure_Fighters_Dataset import main

    main()
    return


def load_to_postgres_venv():
    sys.path.append("/opt/airflow/processing")
    from load_to_postgres import main

    main()
    return

def get_full_warehouse_loading_task_flow():
    start_task = EmptyOperator(task_id="start")

    raw_tables = ["raw_fight_stats", "raw_fighters_current_stats"]
    dim_tables = ["dim_date", "dim_fight_time_format", "dim_fighter",
                  "dim_gender", "dim_method", "dim_result", "dim_weight_class"]
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

def get_incr_warehouse_loading_task_flow():
    start_task = EmptyOperator(task_id="start")

    raw_tables = ["raw_fight_stats", "raw_fighters_current_stats"]
    dim_tables = ["dim_fight_time_format", # "dim_fighter", "dim_date",
                  "dim_gender", "dim_method", "dim_result", "dim_weight_class"]
    fact_tables = [] # "fact_fight"]

    dummy_raw_loading_task = EmptyOperator(task_id="finalize_raw_loading")
    dummy_dim_loading_task = EmptyOperator(task_id="finalize_dim_loading")

    raw_loading_tasks = [
        start_task >>
        SQLExecuteQueryOperator(
            task_id=f"load_{raw_table}",
            conn_id="warehouse_db",
            sql=read_query_from_sql_file(
                os.path.join(PATH_TO_DB_TABLES_INCR_LOADING_SCRIPTS,
                             "raw", f"{raw_table}.sql")
            )
        ) >>
        dummy_raw_loading_task for raw_table in raw_tables
    ]

    dim_loading_tasks  = [
        dummy_raw_loading_task >>
        SQLExecuteQueryOperator(
            task_id=f"load_{dim_table}",
            conn_id="warehouse_db",
            sql=read_query_from_sql_file(
                os.path.join(PATH_TO_DB_TABLES_INCR_LOADING_SCRIPTS,
                             "dimensions", f"{dim_table}.sql")
            )
        ) >>
        dummy_dim_loading_task for dim_table in dim_tables
    ]

    dim_loading_tasks.append(
        dummy_raw_loading_task >>
        SQLExecuteQueryOperator(
            task_id=f"create_dim_fighter",
            conn_id="warehouse_db",
            sql=read_query_from_sql_file(
                os.path.join(PATH_TO_DB_TABLES_CREATION_SCRIPTS,
                             "dimensions", f"dim_fighter.sql")
            )
        ) >>
        SQLExecuteQueryOperator(
            task_id=f"load_dim_fighter",
            conn_id="warehouse_db",
            sql=read_query_from_sql_file(
                os.path.join(PATH_TO_DB_TABLES_FULL_LOADING_SCRIPTS,
                             "dimensions", f"dim_fighter.sql")
            )
        ) >>
        dummy_dim_loading_task
    )

    fact_loading_tasks = [
        dummy_dim_loading_task >>
        SQLExecuteQueryOperator(
            task_id=f"load_{fact_table}",
            conn_id="warehouse_db",
            sql=read_query_from_sql_file(
                os.path.join(PATH_TO_DB_TABLES_INCR_LOADING_SCRIPTS,
                             "facts", f"{fact_table}.sql")
            )
        ) for fact_table in fact_tables
    ]

    fact_loading_tasks.append(
        dummy_dim_loading_task >>
        SQLExecuteQueryOperator(
            task_id=f"create_fact_fight",
            conn_id="warehouse_db",
            sql=read_query_from_sql_file(
                os.path.join(PATH_TO_DB_TABLES_CREATION_SCRIPTS,
                             "facts", f"fact_fight.sql")
            )
        ) >>
        SQLExecuteQueryOperator(
            task_id=f"load_fact_fight",
            conn_id="warehouse_db",
            sql=read_query_from_sql_file(
                os.path.join(PATH_TO_DB_TABLES_FULL_LOADING_SCRIPTS,
                             "facts", f"fact_fight.sql")
            )
        )
    )

    end_task = EmptyOperator(task_id=f"end")

    fact_loading_tasks >> end_task

    return start_task

# TODO: HAVE ADDITIONAL TASKS THAT CHECK THE LOGS IN ORDER TO FIND ANY ERRORS
# THROUGH CRAWLING
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
        circuit_operator = \
            BranchPythonOperator(
                task_id="check_if_incremental_run",
                python_callable=check_if_incremental_or_full_load_run,
                op_kwargs={
                    "is_incremental": "{{ params.is_incremental }}",
                    "incr_task_id": "crawl_data.incremental_crawl_start",
                    "full_task_id": "crawl_data.full_crawl_start"
                }
            )

        full_load_start = get_full_crawl_task_flow()
        incremental_load_start = get_incremental_crawl_task_flow()

        circuit_operator >> [full_load_start, incremental_load_start]

    crawl_join = EmptyOperator(
        task_id="crawl_join",
        trigger_rule="none_failed_min_one_success",
    )

    with TaskGroup(group_id="initial_processing_and_loading") as proc_and_load_group:
        processing_requirements = get_processing_requirements()

        clean_and_process_fights_task = PythonVirtualenvOperator(
            task_id="clean_and_process_fights",
            python_callable=clean_and_process_fights_venv,
            requirements=processing_requirements,
            system_site_packages=False,
        )

        configure_fighters_dataset_task = PythonVirtualenvOperator(
            task_id="configure_fighters_dataset",
            python_callable=configure_fighters_dataset_venv,
            requirements=processing_requirements,
            system_site_packages=False,
        )

        load_to_postgres_task = PythonVirtualenvOperator(
            task_id="load_to_postgres",
            python_callable=load_to_postgres_venv,
            requirements=processing_requirements,
            system_site_packages=False,
        )

        [clean_and_process_fights_task, configure_fighters_dataset_task] >> load_to_postgres_task

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
