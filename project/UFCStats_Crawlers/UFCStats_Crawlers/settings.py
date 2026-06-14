# -*- coding: utf-8 -*-

# Scrapy settings for UFCStats_Crawlers project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://doc.scrapy.org/en/latest/topics/settings.html
#     https://doc.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://doc.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'UFCStats_Crawlers'

SPIDER_MODULES = ['UFCStats_Crawlers.spiders']
NEWSPIDER_MODULE = 'UFCStats_Crawlers.spiders'


# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'UFCStats_Crawlers (+http://www.yourdomain.com)'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy (default: 16)
#CONCURRENT_REQUESTS = 32

# Configure a delay for requests for the same website (default: 0)
# See https://doc.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
DOWNLOAD_DELAY = 0
# The download delay setting will honor only one of:
PLAYWRIGHT_MAX_CONTEXTS = 2

CONCURRENT_REQUESTS = 4
# CONCURRENT_REQUESTS_PER_DOMAIN = 8
#CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
#DEFAULT_REQUEST_HEADERS = {
#   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#   'Accept-Language': 'en',
#}

# Enable or disable spider middlewares
# See https://doc.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    'UFCStats_Crawlers.middlewares.UFCStatsCrawlersSpiderMiddleware': 543,
#}

PLAYWRIGHT_LAUNCH_OPTIONS = {
    "headless": True,
    "args": [
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
    ],
}

DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

def should_abort_request(request):
    url = request.url
    resource_type = request.resource_type

    # Always keep main documents
    if resource_type == "document":
        return False

    # Keep UFCStats browser-check request
    if "ufcstats.com/__c" in url:
        return False

    # Kill visual assets
    if resource_type in {"image", "font", "stylesheet", "media"}:
        return True

    # Kill analytics/tracking
    if any(domain in url for domain in [
        "google-analytics.com",
        "googletagmanager.com",
        "region1.google-analytics.com",
    ]):
        return True

    # Probably safe to kill after testing
    if any(url_part in url for url_part in [
        "code.jquery.com",
        "ajax.googleapis.com/ajax/libs/jquery",
        "ufcstats.com/js/vendor/",
        "ufcstats.com/js/plugins.js",
        "ufcstats.com/js/main.js",
    ]):
        return True

    return False

PLAYWRIGHT_ABORT_REQUEST = should_abort_request

PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 30_000
PLAYWRIGHT_RESTART_DISCONNECTED_BROWSER = True

# Enable or disable downloader middlewares
# See https://doc.scrapy.org/en/latest/topics/downloader-middleware.html
DOWNLOADER_MIDDLEWARES = {
    'UFCStats_Crawlers.middlewares.UFCStatsCrawlersDownloaderMiddleware': 543,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 543
}

# Enable or disable extensions
# See https://doc.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Configure item pipelines
# See https://doc.scrapy.org/en/latest/topics/item-pipeline.html
#ITEM_PIPELINES = {
#    'UFCStats_Crawlers.pipelines.UFCStatsCrawlersPipeline': 300,
#}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://doc.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://doc.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = 'httpcache'
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
HTTPERROR_ALLOW_ALL  = True
RETRY_ENABLED = True
RETRY_TIMES = 8
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 404, 429, 400]

from scrapy.settings.default_settings import RETRY_EXCEPTIONS as DEFAULT_RETRY_EXCEPTIONS

RETRY_EXCEPTIONS = list(DEFAULT_RETRY_EXCEPTIONS) + [
    "playwright._impl._errors.TimeoutError",
]
