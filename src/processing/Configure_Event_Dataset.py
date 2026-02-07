# NOTE: If the fight is the first of a fighter's the resulting stats will be the ones that can be deduced
# from his first fight in order to not have 0, even though this might change. Also I calculate only the
# UFC stats because also the site does not take into account the previous fights.

import os
import pandas as pd
from math import isnan

from utils import get_minio_client, read_csv_from_minio_to_pandas
from utils import write_pandas_csv_to_minio

from Event_Feature_Extractors import get_age_at_time_of_fight, get_avg_fight_time
from Event_Feature_Extractors import get_record_at_time_of_fight, get_sign_strikes_landed_per_min
from Event_Feature_Extractors import get_striking_accuracy, get_sign_strikes_absorbed_per_min
from Event_Feature_Extractors import get_defense, get_takedowns_landed_per_15_mins
from Event_Feature_Extractors import get_takedown_accuracy, get_takedown_defense
from Event_Feature_Extractors import get_submission_attempts_per_15_mins
from Event_Feature_Extractors import get_fighter_height, get_fighter_reach, get_fighter_stance


event_df_common_columns  = ['Fight_ID', 'Fight Date', 'Gender', 'Weight Class', 'Title Fight',
                            'Result', 'Method', 'Round', 'Time', 'Duration_Mins',
                            'Fight Time Format']

event_df_fighter_columns = ['ID', 'Name', 'Nickname', 'Knock Downs', 'Sign.Strikes Done',
                            'Sign.Strikes Attempted', 'Sign.Strikes Perc.','Total Strikes Done',
                            'Total Strikes Attempted', 'Takedowns Done', 'Takedowns Attempted',
                            'Takedowns Perc.', 'Submission Attempts', 'Rev', 'Control']

event_df_fighter_1_prefix = 'Fighter 1 '
event_df_fighter_2_prefix = 'Fighter 2 '

result_df_fighter_1_prefix = 'Fighter_1_'
result_df_fighter_2_prefix = 'Fighter_2_'



def retrieve_initial_dfs(event_df_file_name, fighters_df_file_name):
    minio_client = get_minio_client()

    bucket_name = os.environ.get('MINIO_RAW_DATA_BUCKET_NAME')

    init_event_df = read_csv_from_minio_to_pandas(minio_client, bucket_name,
                                                  event_df_file_name, sep='|',
                                                  header=0)

    fighters_df = read_csv_from_minio_to_pandas(minio_client, bucket_name,
                                                fighters_df_file_name, sep='|',
                                                header=0)
    return init_event_df, fighters_df


def double_events_df(init_event_df):
    event_common_columns = event_df_common_columns

    event_df_fighter_1_columns = [f'{event_df_fighter_1_prefix}{col_name}'
                                  for col_name in event_df_fighter_columns]

    event_df_fighter_2_columns = [f'{event_df_fighter_2_prefix}{col_name}'
                                  for col_name in event_df_fighter_columns]

    # Now we want to have a dataset in a logic where in each row we will have
    # the fighter's stats and his opponents stats, thus, we will double the dataset
    # We also need to switch the result in the "switched" dataset.
    fighter_1_fight_df = init_event_df[event_common_columns +
                                       event_df_fighter_1_columns +
                                       event_df_fighter_2_columns]

    fighter_1_fight_df['is_original_row'] = True

    # Swap Fighter 1 Stats and Fighter 2 Stats in a new row. This means that we
    # also need to reverse the result because it is in the perspective of
    # Fighter 1.
    fighter_2_fight_df = init_event_df[event_common_columns +
                                       event_df_fighter_2_columns +
                                       event_df_fighter_1_columns]

    fighter_2_fight_df['Result'] = fighter_2_fight_df['Result'].apply(lambda x: 'lose' if x == 'win' else
                                                                      ('win' if x == 'lose' else x))

    # Identifier that we will use later.
    fighter_2_fight_df['is_original_row'] = False

    fighter_2_fight_df.columns = fighter_1_fight_df.columns

    fighter_fight_df = pd.concat([fighter_1_fight_df, fighter_2_fight_df],
                                ignore_index=True)

    # Sort by fight Date and Fight ID in order to have the rows in order
    fighter_fight_df = fighter_fight_df.sort_values(by=['Fight Date', 'Fight_ID']).reset_index(drop=True)

    return fighter_fight_df



class FeatureExtractor:
    def __init__(self, name, fn, kwargs=None):
        self.name = name
        self.fn = fn
        self.kwargs = kwargs

        return



    def get_name(self):
        return self.name



    def extract(self):
        return self.fn(**self.kwargs)


def extract_new_dataset_common_features(fighter_fight_df):
    new_df = pd.DataFrame()

    common_features_extractors = [
        FeatureExtractor('Fight_ID',          lambda df: df['Fight_ID'],          {'df': fighter_fight_df}),
        # FeatureExtractor('Fight_Date',        lambda df: df['Fight Date'],        {'df': fighter_fight_df}),
        # FeatureExtractor('Gender',            lambda df: df['Gender'],            {'df': fighter_fight_df}),
        # FeatureExtractor('Weight_Class',      lambda df: df['Weight Class'],      {'df': fighter_fight_df}),
        # FeatureExtractor('Title_Fight',       lambda df: df['Title Fight'],       {'df': fighter_fight_df}),
        # FeatureExtractor('Result',            lambda df: df['Result'],            {'df': fighter_fight_df}),
        # FeatureExtractor('Method',            lambda df: df['Method'],            {'df': fighter_fight_df}),
        # FeatureExtractor('Round',             lambda df: df['Round'],             {'df': fighter_fight_df}),
        # FeatureExtractor('Time',              lambda df: df['Time'],              {'df': fighter_fight_df}),
        # FeatureExtractor('Duration_Mins',     lambda df: df['Duration_Mins'],     {'df': fighter_fight_df}),
        # FeatureExtractor('Fight_Time_Format', lambda df: df['Fight Time Format'], {'df': fighter_fight_df}),
    ]

    for extractor in common_features_extractors:
        new_df[extractor.get_name()] = extractor.extract()

    return new_df



def extract_new_dataset_before_fight_features(fighters_df, fighter_fight_df):
    result_df_fighter_columns = ['ID', 'Name', 'Nickname', 'Age', 'Wins', 'Loses', 'Draws',
                                'Avg_Time(MINS)', 'Height', 'Reach', 'Stance', 'Sign_SLpMin',
                                'Str_Acc', 'Sign_SApMin', 'Defense', 'Takedown_Avgp15M',
                                'Takedown_Acc', 'Takedown_Def', 'Sub_Avgp15M']

    new_df = pd.DataFrame()

    default_kwargs = {'fighters_df': fighters_df, 'fighters_fights_df': fighter_fight_df}

    # TODO: MAYBE REACH, HEIGHT AND STANCE ARE NOT NEEDED
    fighter_features_extractors = [
        FeatureExtractor('ID',                       lambda df: df[f'{event_df_fighter_1_prefix}ID'],       {'df': fighter_fight_df}),
        FeatureExtractor('Name',                     lambda df: df[f'{event_df_fighter_1_prefix}Name'],     {'df': fighter_fight_df}),
        FeatureExtractor('Nickname',                 lambda df: df[f'{event_df_fighter_1_prefix}Nickname'], {'df': fighter_fight_df}),
        FeatureExtractor('Age',                      get_age_at_time_of_fight,                              default_kwargs),
        FeatureExtractor(['Wins', 'Loses', 'Draws'], get_record_at_time_of_fight,                           default_kwargs),
        FeatureExtractor('Avg_Time(MINS)',           get_avg_fight_time,                                    {'fighters_fights_df': fighter_fight_df}),
        # FeatureExtractor('Height',                   get_fighter_height,                                    default_kwargs),
        # FeatureExtractor('Reach',                    get_fighter_reach,                                     default_kwargs),
        # FeatureExtractor('Stance',                   get_fighter_stance,                                    default_kwargs),
        FeatureExtractor('Sign_SLpMin',              get_sign_strikes_landed_per_min,                       {'fighters_fights_df': fighter_fight_df}),
        FeatureExtractor('Str_Acc',                  get_striking_accuracy,                                 {'fighters_fights_df': fighter_fight_df}),
        FeatureExtractor('Sign_SApMin',              get_sign_strikes_absorbed_per_min,                     {'fighters_fights_df': fighter_fight_df}),
        FeatureExtractor('Defense',                  get_defense,                                           {'fighters_fights_df': fighter_fight_df}),
        FeatureExtractor('Takedown_Avgp15M',         get_takedowns_landed_per_15_mins,                      {'fighters_fights_df': fighter_fight_df}),
        FeatureExtractor('Takedown_Acc',             get_takedown_accuracy,                                 {'fighters_fights_df': fighter_fight_df}),
        FeatureExtractor('Takedown_Def',             get_takedown_defense,                                  {'fighters_fights_df': fighter_fight_df}),
        FeatureExtractor('Sub_Avgp15M',              get_submission_attempts_per_15_mins,                   {'fighters_fights_df': fighter_fight_df})
    ]

    for extractor in fighter_features_extractors:
        new_df[extractor.get_name()] = extractor.extract()

    return new_df



def get_final_df(new_common_fight_features_df, fighter_before_fight_features_df):
    # Now we need to split fighter_before_fight_features_df to original and not
    # original rows. The former will constitute the "Fighter 1" and the latter
    # the "Fighter 2".
    fighter_1_before_fight_features_df = fighter_before_fight_features_df[new_common_fight_features_df['is_original_row']]
    fighter_2_before_fight_features_df = fighter_before_fight_features_df[~new_common_fight_features_df['is_original_row']]

    # Resetting the indexes in both dataframes in order to concatenate without problems
    fighter_1_before_fight_features_df = fighter_1_before_fight_features_df.reset_index(drop=True)
    fighter_2_before_fight_features_df = fighter_2_before_fight_features_df.reset_index(drop=True)

    # Renaming the columns with the appropriate prefix
    fighter_1_before_fight_features_df.columns = \
        [f'{result_df_fighter_1_prefix}{col}' for col in fighter_1_before_fight_features_df.columns]

    fighter_2_before_fight_features_df.columns = \
        [f'{result_df_fighter_2_prefix}{col}' for col in fighter_2_before_fight_features_df.columns]

    # Keep the original rows in the common features df and concat the dataframes
    new_common_fight_features_df = new_common_fight_features_df[new_common_fight_features_df['is_original_row']]
    new_common_fight_features_df = new_common_fight_features_df.drop('is_original_row', axis=1).reset_index(drop=True)

    result_df = pd.concat([new_common_fight_features_df, fighter_1_before_fight_features_df, fighter_2_before_fight_features_df],
                          axis=1)

    return result_df


def write_resulting_csv(resulting_df):
    minio_client = get_minio_client()

    bucket_name = os.environ.get('MINIO_RAW_DATA_BUCKET_NAME')

    write_pandas_csv_to_minio(minio_client, bucket_name,
                              "fighter_stats_before_fight.csv",
                              resulting_df, sep='|',
                              index=False)

    return


def main():
    init_event_datafile_name = 'fight_actual_stats_processed.csv'
    init_fighters_datafile_name =  'fighters_current_stats_processed.csv'

    init_event_df, fighters_df = retrieve_initial_dfs(init_event_datafile_name, init_fighters_datafile_name)

    # Now we want to have a dataset in a logic where in each row we will have
    # the fighter's stats and his opponents stats, thus, we will double the dataset
    # We also need to switch the result in the "switched" dataset.

    fighter_fight_df = double_events_df(init_event_df)

    # Get the common features per fight
    new_common_fight_features_df = extract_new_dataset_common_features(fighter_fight_df)
    new_common_fight_features_df['is_original_row'] = fighter_fight_df['is_original_row']

    fighter_before_fight_features_df = extract_new_dataset_before_fight_features(fighters_df, fighter_fight_df)

    print(new_common_fight_features_df)

    print(fighter_before_fight_features_df)

    result_df = get_final_df(new_common_fight_features_df, fighter_before_fight_features_df)

    write_resulting_csv(result_df)

    return 0



if __name__ == "__main__":
    main()