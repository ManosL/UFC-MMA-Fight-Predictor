import os
from minio import Minio
import json


class CrawlingException(Exception):
    pass

def get_minio_client():
    minio_client = Minio(
        'minio:9000',
        access_key=os.environ.get('MINIO_USERNAME'),
        secret_key=os.environ.get('MINIO_PASSWORD'),
        secure=False
    )

    return minio_client

def read_json_from_minio(minio_client, bucket_name, file_name):
    response = minio_client.get_object(bucket_name, file_name)

    # read into pandas
    json_obj = json.loads(response.read().decode("utf-8"))

    # close the response
    response.close()
    response.release_conn()

    return json_obj

def validate_events_crawl(stats):
    log_file_msg = f"Check logs at {stats['log_file_url']}"
    if stats["events_expected"] != stats["events_parsed"]:
        raise CrawlingException(f"Expected to crawl {stats['events_expected']} events but crawled {stats['events_parsed']}.\n{log_file_msg}")

    if stats["fights_expected"] != stats["fights_parsed"]:
        raise CrawlingException(f"Expected to crawl {stats['events_expected']} fights but crawled {stats['events_parsed']}.\n{log_file_msg}")

    if stats["fights_expected"] != stats["item_scraped_count"]:
        raise CrawlingException(f"Expected to scrape {stats['events_expected']} items but scraped {stats['item_scraped_count']}.\n{log_file_msg}")

    return

def validate_fighters_crawl(stats):
    log_file_msg = f"Check logs at {stats['log_file_url']}"
    if stats["fighters_expected"] != stats["fighters_parsed"]:
        raise CrawlingException(f"Expected to crawl {stats['fighters_expected']} fighters but crawled {stats['fighters_parsed']}.\n{log_file_msg}")

    if stats["fighters_expected"] != stats["item_scraped_count"]:
        raise CrawlingException(f"Expected to scrape {stats['fighters_expected']} items but scraped {stats['item_scraped_count']}.\n{log_file_msg}")

    return

def validate_crawl(spider_name, bucket_name, job_id):
    minio_client = get_minio_client()

    stats = read_json_from_minio(minio_client, bucket_name, f"{job_id}.log")

    if spider_name == "event_spider":
        validate_events_crawl(stats)
    elif spider_name == "fighters_spider":
        validate_fighters_crawl(stats)
    else:
        raise ValueError(f"Invalid spider name {spider_name}")

    return
