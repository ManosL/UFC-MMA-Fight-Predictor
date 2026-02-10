import os
import requests
from time import sleep
from datetime import datetime, timezone
import secrets


def schedule_spider(
        base_url,
        project_name,
        spider_name,
        scrapyd_username,
        scrapyd_password,
        is_incremental="False",
        lookup_days=10
):
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H_%M_%S_%f")
    job_id = f"{timestamp}_{secrets.token_hex(3)}"

    response = requests.post(
                    f'{base_url}/schedule.json',
                    data={
                        'project': project_name,
                        'spider': spider_name,
                        'is_incremental': is_incremental,
                        'lookup_days': lookup_days,
                        'jobid': job_id
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

def check_running_spider_job_status(base_url, job_id, scrapyd_username, scrapyd_password):
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
    base_url,
    project_name,
    spider_name,
    is_incremental="False",
    lookup_days=10
):
    scrapyd_username = os.environ.get('SCRAPYD_USERNAME')
    scrapyd_password = os.environ.get('SCRAPYD_PASSWORD')

    base_url = base_url.strip('/')

    response = schedule_spider(base_url, project_name, spider_name,
                               scrapyd_username, scrapyd_password,
                               is_incremental, lookup_days)

    job_id = response['jobid']

    poll_frequency = 5 if is_incremental == "True" else 30
    curr_status = None

    while curr_status != 'finished':
        curr_status = check_running_spider_job_status(base_url, job_id,
                                                      scrapyd_username,
                                                      scrapyd_password
                                                    )
        print(curr_status)
        sleep(poll_frequency)

    print('Crawling finished successfully')
    return job_id

# crawl_from_scrapyd('http://scrapy-server:6800', 'UFCStats_Crawlers', 'event_spider')
