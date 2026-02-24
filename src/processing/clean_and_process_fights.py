import os
import numpy as np
import pandas as pd

from utils import get_minio_client, read_csv_from_minio_to_pandas
from utils import write_pandas_csv_to_minio


event_df_common_columns  = ['Fight Date', 'Gender', 'Weight Class', 'Title Fight',
                            'Result', 'Method', 'Round', 'Time',
                            'Fight Time Format']

event_df_fighter_columns = ['ID', 'Name', 'Nickname', 'Knock Downs', 'Sign.Strikes Done',
                            'Sign.Strikes Attempted', 'Sign.Strikes Perc.','Total Strikes Done',
                            'Total Strikes Attempted', 'Takedowns Done', 'Takedowns Attempted',
                            'Takedowns Perc.', 'Submission Attempts', 'Rev', 'Control']

event_df_fighter_1_prefix = 'Fighter 1 '
event_df_fighter_2_prefix = 'Fighter 2 '


def retrieve_initial_dfs(event_df_file_name):
    minio_client = get_minio_client()

    columns =  event_df_common_columns + \
        [f'{event_df_fighter_1_prefix}{col_name}'
         for col_name in event_df_fighter_columns]

    columns += [f'{event_df_fighter_2_prefix}{col_name}'
                for col_name in event_df_fighter_columns]

    bucket_name = os.environ.get('MINIO_RAW_DATA_BUCKET_NAME')

    init_event_df = read_csv_from_minio_to_pandas(minio_client, bucket_name,
                                                  event_df_file_name, sep='|',
                                                  header=0)

    init_event_df.columns = columns

    return init_event_df



def clean_and_preprocess_initial_dfs(init_event_df):
    init_event_df['Fight Date'] = pd.to_datetime(init_event_df['Fight Date'])

    # I will only keep the matches that are 3 or 5 rounds of 5 minutes because fighters
    # of other fight formats had probably retired and because the requirements to win
    # might be different because at that times the rules were another for example.

    valid_fight_formats = ['3Rnd(5-5-5)', '5Rnd(5-5-5-5-5)', '3Rnd+OT(5-5-5-5)']
    init_event_df = init_event_df[init_event_df['Fight Time Format'].isin(valid_fight_formats)]
    init_event_df = init_event_df.sort_values(by='Fight Date')

    # Add the following here for now but normally it should be on preprocessing
    init_event_df['Duration_Mins'] = init_event_df[['Round', 'Time']].apply(
        lambda x: (int(x['Round']) - 1) * 5.0 + (int(x['Time'].split(':')[0]) +
                                                int(x['Time'].split(':')[1]) / 60),
        axis = 1
    )

    event_df_common_columns.append('Duration_Mins')

    init_event_df.insert(0, 'Fight_ID', range(1,len(init_event_df) + 1))
    event_df_common_columns.insert(0, 'Fight_ID')

    init_event_df = init_event_df.replace('No Stats', np.nan)
    init_event_df = init_event_df.replace('--', np.nan)
    init_event_df = init_event_df.replace('---', np.nan)

    return init_event_df


def write_resulting_csv(resulting_df):
    minio_client = get_minio_client()

    bucket_name = os.environ.get('MINIO_RAW_DATA_BUCKET_NAME')
    write_pandas_csv_to_minio(minio_client, bucket_name,
                              "fight_new_actual_stats_processed.csv",
                              resulting_df, sep='|', na_rep='NaN',
                              index=False)

    return


def main():
    init_event_datafile_name = 'fight_new_actual_stats.csv'

    init_event_df = retrieve_initial_dfs(init_event_datafile_name)
    result_df = clean_and_preprocess_initial_dfs(init_event_df)

    write_resulting_csv(result_df)
    return 0


if __name__ == '__main__':
    main()