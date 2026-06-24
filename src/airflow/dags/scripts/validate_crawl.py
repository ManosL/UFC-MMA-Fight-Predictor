import os
from minio import Minio
from typing import Any
import json

from common.minio_utils import MinioClient


class CrawlingException(Exception):
    pass


def validate_events_crawl(stats: dict[str, Any]) -> None:
    log_file_msg = f"Check logs at {stats['log_file_url']}"
    if stats["events_expected"] != stats["events_parsed"]:
        raise CrawlingException(f"Expected to crawl {stats['events_expected']} events but crawled {stats['events_parsed']}.\n{log_file_msg}")

    if stats["fights_expected"] != stats["fights_parsed"]:
        raise CrawlingException(f"Expected to crawl {stats['events_expected']} fights but crawled {stats['events_parsed']}.\n{log_file_msg}")

    if stats["fights_expected"] != stats["item_scraped_count"]:
        raise CrawlingException(f"Expected to scrape {stats['events_expected']} items but scraped {stats['item_scraped_count']}.\n{log_file_msg}")

    return

def validate_fighters_crawl(stats: dict[str, Any]) -> None:
    log_file_msg = f"Check logs at {stats['log_file_url']}"
    if stats["fighters_expected"] != stats["fighters_parsed"]:
        raise CrawlingException(f"Expected to crawl {stats['fighters_expected']} fighters but crawled {stats['fighters_parsed']}.\n{log_file_msg}")

    if stats["fighters_expected"] != stats["item_scraped_count"]:
        raise CrawlingException(f"Expected to scrape {stats['fighters_expected']} items but scraped {stats['item_scraped_count']}.\n{log_file_msg}")

    return

def validate_crawl(
    spider_name: str,
    bucket_name: str,
    version_id: str
) -> None:
    minio_client = MinioClient(
        'minio:9000',
        access_key=os.environ.get('MINIO_USERNAME'),
        secret_key=os.environ.get('MINIO_PASSWORD'),
        secure=False
    )

    stats = minio_client.read_json_from_minio(bucket_name, f"{version_id}.log")

    if spider_name == "event_spider":
        validate_events_crawl(stats)
    elif spider_name == "fighters_spider":
        validate_fighters_crawl(stats)
    else:
        raise ValueError(f"Invalid spider name {spider_name}")

    return
