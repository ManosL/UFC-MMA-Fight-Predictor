from pathlib import Path
from urllib.parse import urljoin


def log_path_to_scrapyd_url(log_path: str) -> str:
    LOGS_DIR = Path("/app/scrapyd/logs")
    SCRAPYD_BASE_URL = "http://localhost:6800/"

    p = Path(log_path).resolve()
    rel = p.relative_to(LOGS_DIR)  # raises if path is outside logs_dir

    return urljoin(SCRAPYD_BASE_URL, f"logs/{rel.as_posix()}")
