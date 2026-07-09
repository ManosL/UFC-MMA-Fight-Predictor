import json
from minio import Minio
from io import BytesIO
import pandas as pd
from typing import Any


class MinioClient(Minio):
    def create_bucket(self: Minio, bucket_name: str) -> str:
        bucket_exists = self.bucket_exists(bucket_name)

        message = f'Minio Bucket {bucket_name} created successfully.'

        if not bucket_exists:
            self.make_bucket(bucket_name)
        else:
            message = f'Minio Bucket {bucket_name} already exists.'

        return message


    def list_bucket_directory(self: Minio, bucket_name: str, dir_path: str) -> list[Any]:
        return self.list_objects(
            bucket_name, 
            recursive=True, 
            prefix=dir_path
        )


    def read_csv_to_pandas(
        self: Minio,
        bucket_name: str,
        file_name: str,
        **read_csv_kwargs: Any
    ) -> pd.DataFrame:
        response = self.get_object(bucket_name, file_name)

        # read into pandas
        df = pd.read_csv(BytesIO(response.read()), **read_csv_kwargs)

        # close the response
        response.close()
        response.release_conn()

        return df


    def read_json_from_minio(
        self: Minio,
        bucket_name: str,
        file_name: str
    ) -> dict[str, Any]:
        response = self.get_object(bucket_name, file_name)

        json_obj = json.loads(response.read().decode("utf-8"))

        # close the response
        response.close()
        response.release_conn()

        return json_obj


    def write_pandas_df_as_csv(
        self: Minio,
        bucket_name: str,
        file_name: str,
        df: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
        **to_csv_kwargs: Any
    ) -> None:
        csv_buffer = BytesIO()

        df.to_csv(csv_buffer, **to_csv_kwargs)
        csv_buffer.seek(0)  # reset pointer to start

        # Upload to MinIO
        self.put_object(
            bucket_name,
            file_name,
            data=csv_buffer,
            length=len(csv_buffer.getvalue()),
            content_type="application/csv",
            metadata=metadata
        )

        return


    def write_json(
        self: Minio,
        bucket_name: str,
        file_name: str,
        json_obj: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        json_buffer = BytesIO(json.dumps(json_obj).encode("utf-8"))

        json_buffer.seek(0)  # reset pointer to start

        # Upload to MinIO
        self.put_object(
            bucket_name,
            file_name,
            data=json_buffer,
            length=len(json_buffer.getvalue()),
            content_type="application/json",
            metadata=metadata
        )

        return

    def write_file(
        self: Minio,
        bucket_name: str,
        src_file_name: str,
        dst_file_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.fput_object(
            bucket_name,
            dst_file_name,
            src_file_name,
            metadata=metadata
        )

        return