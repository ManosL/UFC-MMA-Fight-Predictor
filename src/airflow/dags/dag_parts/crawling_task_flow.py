import os
import sys
from airflow.models.xcom_arg import XComArg
from airflow.models.baseoperator import BaseOperator
from airflow.operators.python import PythonVirtualenvOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator


def crawl_from_scrapyd_venv(
    base_url: str,
    project_name: str,
    spider_name: str,
    is_incremental: bool = "False",
    lookup_days: int = 15,
) -> str:
    sys.path.append('/opt/airflow/dags')
    from scripts.crawl_from_scrapyd import crawl_from_scrapyd

    job_id = crawl_from_scrapyd(
        base_url,
        project_name,
        spider_name,
        is_incremental,
        int(lookup_days)
    )

    return job_id


def validate_crawl_venv(
    spider_name: str,
    bucket_name: str,
    job_id: str
) -> None:
    sys.path.append('/opt/airflow/dags')
    from scripts.validate_crawl import validate_crawl

    validate_crawl(spider_name, bucket_name, job_id)

    return


def check_if_incremental_or_full_load_run(
    is_incremental: str,
    incr_task_id: str,
    full_task_id: str
) -> str:
    return incr_task_id if is_incremental.lower() == "true" else full_task_id


def get_full_crawl_task_flow() -> BaseOperator:
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

    validate_fights_crawl_task = PythonVirtualenvOperator(
        task_id="full_crawl_fights_validate",
        python_callable=validate_crawl_venv,
        op_kwargs={
            "spider_name": "event_spider",
            "bucket_name": os.environ["MINIO_EVENT_CRAWL_LOGS_BUCKET_NAME"],
            "job_id": XComArg(crawl_fights_task)
        },
        requirements=[
            "minio",
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

    validate_fighters_crawl_task = PythonVirtualenvOperator(
        task_id="full_crawl_fighters_validate",
        python_callable=validate_crawl_venv,
        op_kwargs={
            "spider_name": "fighters_spider",
            "bucket_name": os.environ["MINIO_FIGHTER_CRAWL_LOGS_BUCKET_NAME"],
            "job_id": XComArg(crawl_fighters_task)
        },
        requirements=[
            "minio",
        ],
        system_site_packages=False,  # important: isolate from Airflow deps
    )

    end_task = EmptyOperator(task_id="full_crawl_end")

    start_task >> [crawl_fights_task, crawl_fighters_task]
    crawl_fights_task >> validate_fights_crawl_task
    crawl_fighters_task >> validate_fighters_crawl_task
    [validate_fights_crawl_task, validate_fighters_crawl_task] >> end_task

    return start_task


def get_incremental_crawl_task_flow() -> BaseOperator:
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

    validate_fights_crawl_task = PythonVirtualenvOperator(
        task_id="incremental_fights_validate",
        python_callable=validate_crawl_venv,
        op_kwargs={
            "spider_name": "event_spider",
            "bucket_name": os.environ["MINIO_EVENT_CRAWL_LOGS_BUCKET_NAME"],
            "job_id": XComArg(crawl_fights_task)
        },
        requirements=[
            "minio",
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

    validate_fighters_crawl_task = PythonVirtualenvOperator(
        task_id="incremental_fighters_validate",
        python_callable=validate_crawl_venv,
        op_kwargs={
            "spider_name": "fighters_spider",
            "bucket_name": os.environ["MINIO_FIGHTER_CRAWL_LOGS_BUCKET_NAME"],
            "job_id": XComArg(crawl_fighters_task)
        },
        requirements=[
            "minio",
        ],
        system_site_packages=False,  # important: isolate from Airflow deps
    )

    end_task = EmptyOperator(task_id="incremental_crawl_end")

    start_task >> crawl_fights_task >> validate_fights_crawl_task
    validate_fights_crawl_task >> crawl_fighters_task
    crawl_fighters_task >> validate_fighters_crawl_task >>end_task
    return start_task

def get_crawling_task_flow() -> BaseOperator:
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

    return circuit_operator
