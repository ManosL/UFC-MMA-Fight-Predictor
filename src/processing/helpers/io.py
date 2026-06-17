import os
import pandas as pd
from typing import Any

from common.minio_utils import MinioClient


def get_minio_client() -> MinioClient:
    return MinioClient(
        'minio:9000',
        access_key=os.environ.get('MINIO_USERNAME'),
        secret_key=os.environ.get('MINIO_PASSWORD'),
        secure=False
    )


def get_minio_bucket_name() -> str:
    return os.environ.get('MINIO_RAW_DATA_BUCKET_NAME')


def retrieve_df_from_csv(
    filename: str,
    extra_read_kwargs: dict[str, Any] | None = None
) -> pd.DataFrame:
    minio_client = get_minio_client()

    bucket_name = get_minio_bucket_name()

    read_kwargs = {
        "sep": "|",
        "header": 0
    }

    if extra_read_kwargs:
        read_kwargs = read_kwargs | extra_read_kwargs

    df = minio_client.read_csv_to_pandas(
        bucket_name,
        filename,
        **read_kwargs
    )

    return df


def write_resulting_csv(
    resulting_df: pd.DataFrame,
    filename: str
) -> None:
    minio_client = get_minio_client()

    bucket_name = get_minio_bucket_name()

    minio_client.write_pandas_df_as_csv(
        bucket_name,
        filename,
        resulting_df,
        sep='|',
        na_rep='NaN',
        index=False
    )

    return
