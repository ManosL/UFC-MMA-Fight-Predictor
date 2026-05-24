import os
import requests
from time import sleep
from datetime import datetime, timezone
import secrets
from typing import Any


def schedule_spider(
        base_url: str,
        project_name: str,
        spider_name: str,
        scrapyd_username: str,
        scrapyd_password: str,
        version_id: str,
        is_incremental: str = "False",
        lookup_days: int = 10,
) -> dict[str, Any]:
    response = requests.post(
                    f'{base_url}/schedule.json',
                    data={
                        'project': project_name,
                        'spider': spider_name,
                        'is_incremental': is_incremental,
                        'lookup_days': lookup_days,
                        'jobid': f"{spider_name}_{version_id}"
                    },
                    auth=(scrapyd_username, scrapyd_password,),
                )

    if response.status_code != 200:
        print(f'schedule.json gave {response.status_code}')
        response.raise_for_status()

    response = response.json()

    if response['status'] != 'ok':
        raise ValueError(f'Got invalid status when scheduling the spider: {response["status"]}')

    return response

def check_running_spider_job_status(
    base_url: str,
    job_id: str,
    scrapyd_username: str,
    scrapyd_password: str
) -> str:
    response = requests.get(
                    f'{base_url}/status.json',
                    params={'job': job_id},
                    auth=(scrapyd_username, scrapyd_password),
                )

    if response.status_code != 200:
        print(f'status.json gave {response.status_code}')
        response.raise_for_status()

    response = response.json()
    return response['currstate']


def crawl_from_scrapyd(
    base_url: str,
    project_name: str,
    spider_name: str,
    version_id: str,
    is_incremental: str = "False",
    lookup_days: int = 10
) -> str:
    scrapyd_username = os.environ.get('SCRAPYD_USERNAME')
    scrapyd_password = os.environ.get('SCRAPYD_PASSWORD')

    base_url = base_url.strip('/')

    response = schedule_spider(
        base_url, 
        project_name, 
        spider_name,
        scrapyd_username, 
        scrapyd_password,
        version_id,
        is_incremental, 
        lookup_days
    )

    job_id = response['jobid']

    poll_frequency = 5 if is_incremental == "True" else 30
    curr_status = None

    while curr_status != 'finished':
        curr_status = check_running_spider_job_status(
            base_url, 
            job_id,
            scrapyd_username,
            scrapyd_password
        )
        print(curr_status)
        sleep(poll_frequency)

    print('Crawling finished successfully')
    return version_id
