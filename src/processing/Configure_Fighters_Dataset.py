import os
import re
import pandas as pd
import numpy  as np

from utils import get_minio_client, read_csv_from_minio_to_pandas
from utils import write_pandas_csv_to_minio

from fighter_unknown_gender_map import full_names_to_gender
import gender_guesser.detector as gender


def retrieve_initial_dfs(event_df_file_name, fighters_df_file_name):
    minio_client = get_minio_client()

    bucket_name = os.environ.get('MINIO_RAW_DATA_BUCKET_NAME')

    init_event_df = read_csv_from_minio_to_pandas(minio_client, bucket_name,
                                                  event_df_file_name, sep='|',
                                                  header=0)

    fighters_df = read_csv_from_minio_to_pandas(minio_client, bucket_name,
                                                fighters_df_file_name,
                                                sep='|', header=0)
    return init_event_df, fighters_df


def convert_no_stat_to_nan(fighters_df):
    fighters_df = fighters_df.replace('No Stat', np.nan)
    fighters_df = fighters_df.replace('--', np.nan)
    fighters_df = fighters_df.replace('---', np.nan)

    return fighters_df



def convert_percentage_features_to_decimals(fighters_df):
    percentage_features = ['Str.Acc.', 'Str. Def.', 'TD Acc.', 'TD Def.']

    for feature in percentage_features:
        fighters_df[feature] = fighters_df[feature].apply(lambda x: float(x.strip('%')) / 100 if x is not np.nan else np.nan)

    return fighters_df



# Should be an str of form mm:ss
def convert_time_str_to_mins(elem):
    mins = int(elem.split(':')[0])
    secs = int(elem.split(':')[1])

    return_val = (mins * 60 + secs) / 60

    return return_val



# def find_average_fight_time(fights_df, fighters_df):
#     fighter_ids     = list(fighters_df['Fighter ID'])
#     avg_fight_times = []

#     for fighter_id in fighter_ids:
#         fighters_fights_1 = fights_df[(fights_df['Fighter 1 ID'] == fighter_id)]
#         fighters_fights_2 = fights_df[(fights_df['Fighter 2 ID'] == fighter_id)]

#         curr_df = pd.concat([fighters_fights_1, fighters_fights_2], axis=0)

#         fights_rounds_array = np.array(curr_df['Round'])

#         last_round_dur       = curr_df['Time']
#         last_round_dur_array = np.array(last_round_dur.apply(lambda x: convert_time_str_to_mins(x)))

#         fights_duration_array = (fights_rounds_array - 1) * 5 + last_round_dur_array

#         if len(fights_duration_array) == 0:
#             avg_fight_times.append(0)
#         else:
#             avg_fight_times.append(fights_duration_array.mean())

#     fighters_df.insert(5, 'Avg.Time(in Mins)', avg_fight_times)

#     return fighters_df



def find_genders(fights_df, fighters_df):
    d = gender.Detector()

    fighters_ids   = list(fighters_df['Fighter ID'])
    fighters_names = list(fighters_df['Fighter Name'])
    genders = []

    for i in range(len(fighters_ids)):
        fighter_id     = fighters_ids[i]

        fighters_fights_1 = fights_df[(fights_df['Fighter 1 ID'] == fighter_id)]
        fighters_fights_2 = fights_df[(fights_df['Fighter 2 ID'] == fighter_id)]

        # If the fighter did not fought previously we cannot determine its gender
        # from his/her fights
        if len(fighters_fights_1) == 0 and len(fighters_fights_2) == 0:
            # If the fighter does not have any fights, use the detector
            fighter_name = fighters_names[i]

            fighters_gender = d.get_gender(fighter_name.split()[0])

            fighters_gender = "male" if fighters_gender in {"male", 'mostly_male'} else fighters_gender
            fighters_gender = "female" if fighters_gender in {"female", 'mostly_female'} else fighters_gender

            # If the detector cannot determine the gender, our last hope is to use the hard coded dict
            if fighters_gender in {'andy', 'unknown'}:
                if fighter_name in full_names_to_gender.keys():
                    fighters_gender = full_names_to_gender[fighter_name]

            genders.append(fighters_gender)
        else:
            if len(fighters_fights_1) > 0:
                # A special case for catch weight fights because in this case
                # the gender is not captured
                if len(fighters_fights_1['Gender'].unique()) == 2:
                    fighters_gender = 'female'
                else:
                    fighters_gender = list(fighters_fights_1['Gender'])[0]
            else:
                if len(fighters_fights_2['Gender'].unique()) == 2:
                    fighters_gender = 'female'
                else:
                    fighters_gender = list(fighters_fights_2['Gender'])[0]

            genders.append(fighters_gender)

    genders = pd.Series(genders)
    ambiguous = fighters_df[genders.isin(['andy', 'unknown'])]['Fighter Name']

    if len(ambiguous) > 0:
        print('The following fighters cannot have their gender specified, please complete it yourself, in the generated csv file')
        print('\n'.join(list(ambiguous)))
        print('\nYou should fill their genders in the csv or in the dictionary only in the case you want to use them to predict a fight for them.')

    fighters_df.insert(1, 'Gender', genders)

    return fighters_df



def convert_feet_and_inches_to_cm(x):
    if x is np.nan:
        return np.nan

    res = re.match(r"(\d+)\'\s+(\d+)\"", x)

    assert (res.group(1) is not None) and (res.group(2) is not None)

    feet = int(res.group(1))
    inches = int(res.group(2))

    return (feet * 12 + inches) * 2.54


def convert_weight_to_int(x):
    if x is np.nan:
        return np.nan

    res = re.match(r"(\d+)\s+lbs\.", x)

    assert res.group(1) is not None
    return int(res.group(1))


def convert_inches_to_cm(x):
    if x is np.nan or len(x) == 0:
        return np.nan

    res = re.match(r"(\d+)\"", x)

    assert res.group(1) is not None
    inches = int(res.group(1))

    return (inches) * 2.54


def write_resulting_csv(resulting_df):
    minio_client = get_minio_client()

    bucket_name = os.environ.get('MINIO_RAW_DATA_BUCKET_NAME')

    write_pandas_csv_to_minio(minio_client, bucket_name,
                              "fighters_new_current_stats_processed.csv",
                              resulting_df, sep='|',
                              na_rep='NaN', index=False)

    return


def main():
    fights_datafile_path = 'fight_new_actual_stats.csv'
    fighters_datafile_path = 'fighters_new_current_stats.csv'

    # Do the preprocessing steps
    fights_df, fighters_df = retrieve_initial_dfs(fights_datafile_path, fighters_datafile_path)
    fighters_df = convert_no_stat_to_nan(fighters_df)
    fighters_df = convert_percentage_features_to_decimals(fighters_df)
    # fighters_df = find_average_fight_time(fights_df, fighters_df)
    fighters_df = find_genders(fights_df, fighters_df)

    fighters_df['Height'] = fighters_df['Height'].apply(lambda x: convert_feet_and_inches_to_cm(x))
    fighters_df['Weight'] = fighters_df['Weight'].apply(lambda x: convert_weight_to_int(x))
    fighters_df['Reach']  = fighters_df['Reach'].apply(lambda x: convert_inches_to_cm(x))
    fighters_df['Stance'] = fighters_df['Stance'].apply(lambda x: 'Unknown' if x is np.nan else x)

    fighters_df['DOB'] = pd.to_datetime(fighters_df['DOB'])

    write_resulting_csv(fighters_df)

    return 0

if __name__ == "__main__":
    main()