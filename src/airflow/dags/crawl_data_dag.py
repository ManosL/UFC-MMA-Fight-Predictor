from datetime import datetime
import sys
from airflow import DAG
from airflow.sdk import Param, get_current_context
from airflow.operators.python import PythonVirtualenvOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator


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


def check_if_incremental_or_full_load_run(is_incremental):
    print(is_incremental)
    return "incremental_load_start" if is_incremental.lower() == "true" else "full_load_start"


def get_full_load_task_flow():
    start_task = EmptyOperator(task_id="full_load_start")

    crawl_fights_task = PythonVirtualenvOperator(
        task_id="full_load_crawl_fights",
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
        task_id="full_load_crawl_fighters",
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

    end_task = EmptyOperator(task_id="full_load_end")

    start_task >> [crawl_fights_task, crawl_fighters_task] >> end_task
    return start_task


def get_incremental_load_task_flow():
    start_task = EmptyOperator(task_id="incremental_load_start")

    crawl_fights_task = PythonVirtualenvOperator(
        task_id="incremental_load_crawl_fights",
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
        task_id="incremental_load_crawl_fighters",
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

    end_task = EmptyOperator(task_id="incremental_load_end")

    start_task >> crawl_fights_task >> crawl_fighters_task >> end_task
    return start_task


with DAG(
    dag_id="crawl_ufcstats_site",
    params={
        "is_incremental": Param(False, type="boolean"),
        "lookup_days": Param(15, type="integer")
    },
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    circuit_operator = \
        BranchPythonOperator(
            task_id="check_if_incremental_run",
            python_callable=check_if_incremental_or_full_load_run,
            op_kwargs={
                "is_incremental": "{{ params.is_incremental }}"
            }
        )

    full_load_start = get_full_load_task_flow()
    incremental_load_start = get_incremental_load_task_flow()

    circuit_operator >> [full_load_start, incremental_load_start]
