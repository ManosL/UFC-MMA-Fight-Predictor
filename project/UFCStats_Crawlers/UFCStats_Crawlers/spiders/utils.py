from copy import deepcopy
import os
import re
from pathlib import Path
from urllib.parse import urljoin
import scrapy

from common.minio_utils import MinioClient


def get_fighter_id_from_url(url: str) -> str:
    id = re.match(r'^.*ufcstats.com/fighter-details/([a-zA-Z0-9]*)(\/|\?)?.*$', url).groups()[0]
    assert(id != None)

    return id


def get_fight_id_from_url(url: str) -> str:
    id = re.match(r'^.*ufcstats.com/fight-details/([a-zA-Z0-9]*)(\/|\?)?.*$', url).groups()[0]
    assert(id != None)

    return id.strip()
	

def log_path_to_scrapyd_url(log_path: str) -> str:
    LOGS_DIR = Path("/app/scrapyd/logs")
    SCRAPYD_BASE_URL = "http://localhost:6800/"

    p = Path(log_path).resolve()
    rel = p.relative_to(LOGS_DIR)  # raises if path is outside logs_dir

    return urljoin(SCRAPYD_BASE_URL, f"logs/{rel.as_posix()}")


def save_crawling_stats_to_minio(
    spider: scrapy.Spider,
    bucket_name: str
) -> None:
    minio_client = MinioClient(
        'minio:9000',
        access_key=os.environ.get('MINIO_USERNAME'),
        secret_key=os.environ.get('MINIO_PASSWORD'),
        secure=False
    )

    log_file_url = log_path_to_scrapyd_url(spider.settings["LOG_FILE"])

    spider.crawler.stats.set_value("spider_name", spider.name)
    spider.crawler.stats.set_value("log_file_url", log_file_url)

    stats_file_name = os.path.basename(spider.settings["LOG_FILE"])
    stats_file_name = stats_file_name.replace(f"{spider.name}_", "", 1)

    spider.logger.info(minio_client.create_bucket(bucket_name))

    crawl_stats = deepcopy(spider.crawler.stats.get_stats())
    crawl_stats["start_time"] = str(crawl_stats["start_time"])

    minio_client.write_json(
        bucket_name,
        stats_file_name,
        crawl_stats
    )

    return
