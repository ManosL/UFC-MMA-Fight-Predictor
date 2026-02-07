# TODO: THIS FILE REMOVE IT FROM THERE AND CHANGE THE VOLUMES TO BE LINKED WITH IT.
import os
from minio import Minio
from io import BytesIO, StringIO
import pandas as pd


def get_minio_client():
    minio_client = Minio(
        'minio:9000',
        access_key=os.environ.get('MINIO_USERNAME'),
        secret_key=os.environ.get('MINIO_PASSWORD'),
        secure=False
    )

    return minio_client


def read_csv_from_minio_to_pandas(minio_client, bucket_name,
                                  file_name, **read_csv_kwargs):
    response = minio_client.get_object(bucket_name, file_name)

    # read into pandas
    df = pd.read_csv(BytesIO(response.read()), **read_csv_kwargs)

    # close the response
    response.close()
    response.release_conn()

    return df


def write_pandas_csv_to_minio(minio_client, bucket_name,
                              file_name, df,
                              **to_csv_kwargs):
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
