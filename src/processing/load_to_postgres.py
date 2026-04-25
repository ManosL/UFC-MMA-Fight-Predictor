from io import StringIO

from common.psycopg_utils import get_postgres_connection

from constants import (
    PROCESSED_FIGHT_STATS_FILENAME,
    PROCESSED_FIGHTER_STATS_FILENAME
)

from helpers.io import (
    retrieve_df_from_csv
)


def main() -> int:
    postgres_conn = get_postgres_connection()
    postgres_cursor = postgres_conn.cursor()

    files_to_load = [
        PROCESSED_FIGHT_STATS_FILENAME,
        PROCESSED_FIGHTER_STATS_FILENAME,
    ]

    tables_to_load_to = [
        'new_fight_stats',
        'new_fighters_current_stats',
    ]

    for filename, table_name in zip(files_to_load, tables_to_load_to):
        table_creation_file_path = f'/opt/airflow/sql/creation/raw/{table_name}.sql'

        df = retrieve_df_from_csv(filename)

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
    main()
