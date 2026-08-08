import os
import argparse
from io import StringIO

from common.psycopg_utils import get_postgres_connection

from constants import (
    PROCESSED_FIGHT_STATS_FILENAME,
    PROCESSED_FIGHTER_STATS_FILENAME,
    get_fight_df_integer_columns,
)

from processing_helpers.io import (
    retrieve_df_from_csv
)


def main(version_id: str) -> int:
    postgres_conn = get_postgres_connection()
    postgres_cursor = postgres_conn.cursor()

    files_to_load = [
        PROCESSED_FIGHT_STATS_FILENAME,
        PROCESSED_FIGHTER_STATS_FILENAME,
    ]

    files_to_load = [os.path.join(version_id, filename) for filename in files_to_load]

    tables_to_load_to = [
        'new_fight_stats',
        'new_fighters_current_stats',
    ]

    for filename, table_name in zip(files_to_load, tables_to_load_to):
        table_creation_file_path = f'/opt/airflow/sql/creation/raw/{table_name}.sql'

        read_kwargs = {}

        if table_name == "new_fight_stats":
            read_kwargs = {"dtype": {col: "Int64" for col in get_fight_df_integer_columns()}}

        df = retrieve_df_from_csv(filename, read_kwargs)

        print(df.head(10))

        with open(table_creation_file_path, 'r') as table_creation_file:
            table_creation_sql = table_creation_file.read()
            postgres_cursor.execute(table_creation_sql)
            postgres_conn.commit()

        buffer = StringIO()
        df.to_csv(
            buffer,
            sep='|',
            index=False,
            na_rep='NaN',
            header=False
        )  # copy_from doesn’t handle headers

        buffer.seek(0)

        postgres_cursor.copy_from(
            buffer,
            table_name,
            sep="|",
            null="NaN"
        )  # can change sep/null if needed

        postgres_conn.commit()

    postgres_cursor.close()
    postgres_conn.close()

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--version_id", "-v", help="Version ID")
    args = parser.parse_args()
    main(args.version_id)
