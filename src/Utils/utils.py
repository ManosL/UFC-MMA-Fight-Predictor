import pandas as pd

# Fighters dataset does not contain any labels but I
# provide this util in case its needed

def read_fighters_data(fighters_df_path):
    return pd.read_csv(fighters_df_path, sep='|', header=0)

# Fights dataset contains multiple labels(the Result,Method, Round and Time
# columns), thus this functtion will return 2 dataframes. One with the 
# "independent" columns and one with the labels

def read_fights_data(fights_df_path):
    label_columns = ['Result', 'Method', 'Round', 'Time']

    fights_df = pd.read_csv(fights_df_path, sep='|', header=0)

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
