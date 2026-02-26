import os
from minio import Minio
from io import BytesIO, StringIO
import pandas as pd
import psycopg2

from utils import get_minio_client, get_postgres_connection
from utils import read_csv_from_minio_to_pandas


def main() -> int:
    minio_client = get_minio_client()
    postgres_conn = get_postgres_connection()
    postgres_cursor = postgres_conn.cursor()

    bucket_name = os.environ.get('MINIO_RAW_DATA_BUCKET_NAME')

    files_to_load = [
                        'fight_new_actual_stats_processed.csv',
                        'fighters_new_current_stats_processed.csv',
                    ]

    tables_to_load_to = [
                            'new_fight_stats',
                            'new_fighters_current_stats',
                        ]

    for filename, table_name in zip(files_to_load, tables_to_load_to):
        table_creation_file_path = f'/opt/airflow/sql/creation/raw/{table_name}.sql'

        df = read_csv_from_minio_to_pandas(minio_client, bucket_name, filename,
                                           sep='|', header=0)

        print(df.head(10))

        with open(table_creation_file_path, 'r') as table_creation_file:
            table_creation_sql = table_creation_file.read()
            postgres_cursor.execute(table_creation_sql)
            postgres_conn.commit()

        buffer = StringIO()
        df.to_csv(buffer, sep='|', index=False, na_rep='NaN', header=False)  # copy_from doesn’t handle headers
        buffer.seek(0)

        postgres_cursor.copy_from(buffer, table_name, sep="|",
                                  null="NaN")  # can change sep/null if needed
        postgres_conn.commit()

    postgres_cursor.close()
    postgres_conn.close()

    return 0


if __name__ == '__main__':
    main()
