from copy import deepcopy
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
import scrapy
from scrapy_playwright.page import PageMethod

from common.minio_utils import MinioClient


def get_fighter_id_from_url(url: str) -> str:
    id = re.match(r'^.*ufcstats.com/fighter-details/([a-zA-Z0-9]*)(\/|\?)?.*$', url).groups()[0]
    assert(id != None)

    return id


def get_fight_id_from_url(url: str) -> str:
    id = re.match(r'^.*ufcstats.com/fight-details/([a-zA-Z0-9]*)(\/|\?)?.*$', url).groups()[0]
    assert(id != None)

    return id.strip()
	

def get_playwright_kwargs(
    playwright_context: str,
    selectors_to_wait: list[str],
    *,
    include_page: bool = True,
) -> dict[str, Any]:
    return {
        "playwright": True,
        "playwright_include_page": include_page,
        "playwright_context": playwright_context,
        "playwright_page_methods": [
            PageMethod(
                "wait_for_selector", 
                selector,
                state="attached",
            )
            for selector in selectors_to_wait
        ],
        "playwright_page_goto_kwargs": {
            "wait_until": "commit",
        }
    }


async def get_cookies_from_playwright_page(response):
    page = response.meta["playwright_page"]

    playwright_cookies = await page.context.cookies()
    await page.close()

    return {
        cookie["name"]: cookie["value"]
        for cookie in playwright_cookies
        if cookie["domain"] in {"www.ufcstats.com", ".ufcstats.com"}
    }


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

    crawl_stats = deepcopy(spider.crawler.stats.get_stats())
    crawl_stats["start_time"] = str(crawl_stats["start_time"])

    minio_client.write_json(
        bucket_name,
        stats_file_name,
        crawl_stats
    )

    return
