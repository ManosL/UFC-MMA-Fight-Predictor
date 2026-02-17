import os
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2


def get_postgres_connection():
    conn = psycopg2.connect(
        dbname=os.environ.get('DATA_WAREHOUSE_POSTGRES_DB_NAME'),
        user=os.environ.get('DATA_WAREHOUSE_POSTGRES_USERNAME'),
        password=os.environ.get('DATA_WAREHOUSE_POSTGRES_PASSWORD'),
        host='postgres-1',
        port=os.environ.get('DATA_WAREHOUSE_POSTGRES_HOST_PORT'),
    )

    return conn


# Fighters dataset does not contain any labels but I
# provide this util in case its needed

# TODO: USE DECORATORS THAT INITIALIZES AND CLOSES THE CONNECTION
def read_fighters_data(fighters_df_path):
    postgres_connection = get_postgres_connection()
    query = "SELECT * FROM \"ML_Fighters\""

    data = sqlio.read_sql_query(query, postgres_connection)

    postgres_connection.close()
    return data

# Fights dataset contains multiple labels(the Result,Method, Round and Time
# columns), thus this functtion will return 2 dataframes. One with the
# "independent" columns and one with the labels

def read_fights_data(fights_df_path):
    label_columns = ['Result', 'Method', 'Round', 'Time']

    postgres_connection = get_postgres_connection()
    query = "SELECT * FROM \"ML_Fights\""

    fights_df = sqlio.read_sql_query(query, postgres_connection)

    postgres_connection.close()

    # Also writing in labels df the Fight_ID(the IDs in both datasets
    # are in the same order)
    labels    = fights_df[['Fight_ID'] + label_columns]
    attrs     = fights_df.drop(label_columns, axis=1)

    return attrs, labels

def read_matchup_data(matchup_df_path):
    return pd.read_csv(matchup_df_path, sep='|', header=None,
                    names=['Weight Class', 'Title Fight', 'Rounds',
                            'Fighter 1 ID', 'Fighter 2 ID'])

def df_get_na(df):
    return df.isna().sum()
