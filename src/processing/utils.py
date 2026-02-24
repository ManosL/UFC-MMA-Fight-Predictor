import os
from minio import Minio
from io import BytesIO, StringIO
import pandas as pd
import psycopg2
from psycopg2.extensions import connection
from typing import Any


def get_minio_client() -> Minio:
    minio_client = Minio(
        'minio:9000',
        access_key=os.environ.get('MINIO_USERNAME'),
        secret_key=os.environ.get('MINIO_PASSWORD'),
        secure=False
    )

    return minio_client


def get_postgres_connection() -> connection:
    conn = psycopg2.connect(
        dbname=os.environ.get('DATA_WAREHOUSE_POSTGRES_DB_NAME'),
        user=os.environ.get('DATA_WAREHOUSE_POSTGRES_USERNAME'),
        password=os.environ.get('DATA_WAREHOUSE_POSTGRES_PASSWORD'),
        host='postgres-1',
        port=os.environ.get('DATA_WAREHOUSE_POSTGRES_HOST_PORT'),
    )

    return conn


def read_csv_from_minio_to_pandas(
    minio_client: Minio,
    bucket_name: str,
    file_name: str,
    **read_csv_kwargs: Any
) -> pd.DataFrame:
    response = minio_client.get_object(bucket_name, file_name)

    # read into pandas
    df = pd.read_csv(BytesIO(response.read()), **read_csv_kwargs)

    # close the response
    response.close()
    response.release_conn()

    return df


def write_pandas_csv_to_minio(
    minio_client: Minio,
    bucket_name: str,
    file_name: str,
    df: pd.DataFrame,
    **to_csv_kwargs: Any
) -> None:
    csv_buffer = BytesIO()

    df.to_csv(csv_buffer, **to_csv_kwargs)
    csv_buffer.seek(0)  # reset pointer to start

    # Upload to MinIO
    minio_client.put_object(
        bucket_name,
        file_name,
        data=csv_buffer,
        length=len(csv_buffer.getvalue()),
        content_type="application/csv"
    )

    return
