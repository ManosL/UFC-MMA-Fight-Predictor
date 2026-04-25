import os
import psycopg2
from psycopg2.extensions import connection


def get_postgres_connection() -> connection:
    conn = psycopg2.connect(
        dbname=os.environ.get('DATA_WAREHOUSE_POSTGRES_DB_NAME'),
        user=os.environ.get('DATA_WAREHOUSE_POSTGRES_USERNAME'),
        password=os.environ.get('DATA_WAREHOUSE_POSTGRES_PASSWORD'),
        host='postgres-1',
        port=os.environ.get('DATA_WAREHOUSE_POSTGRES_HOST_PORT'),
    )

    return conn
