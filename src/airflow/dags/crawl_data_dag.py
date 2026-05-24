from datetime import datetime
from airflow import DAG
from airflow.sdk import Param

from dag_parts.crawling_task_flow import get_crawling_task_flow

with DAG(
    dag_id="crawl_ufcstats_site",
    params={
        "is_incremental": Param(False, type="boolean"),
        "lookup_days": Param(15, type="integer"),
        "version_id": Param("", type=["string"])
    },
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    get_crawling_task_flow(
        incremental_start_task_id="incremental_crawl_start",
        full_start_task_id="full_crawl_start"
    )
