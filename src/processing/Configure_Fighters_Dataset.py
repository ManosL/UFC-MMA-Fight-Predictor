import pandas as pd

from constants import (
    RAW_FIGHT_STATS_FILENAME,
    RAW_FIGHTER_STATS_FILENAME,
    PROCESSED_FIGHTER_STATS_FILENAME
)

from helpers.gender_resolution import (
    resolve_fighter_gender
)

from helpers.io import (
    retrieve_df_from_csv,
    write_resulting_csv
)

from helpers.transform import (
    normalize_no_val_to_nan,
    convert_percentage_features_to_decimals,
    convert_feet_and_inches_to_cm,
    convert_weight_to_int,
    convert_inches_to_cm
)


def process_raw_fighters_df(
    raw_fighters_df: pd.DataFrame,
    raw_fights_df: pd.DataFrame
) -> pd.DataFrame:
    processed_fighters_df = raw_fighters_df.copy()

    raw_fights_df['Fight Date'] = pd.to_datetime(raw_fights_df['Fight Date'])

    processed_fighters_df = normalize_no_val_to_nan(processed_fighters_df, ["No Stat", "--", "---"])
    processed_fighters_df = convert_percentage_features_to_decimals(
        processed_fighters_df,
        percentage_features=['Str.Acc.', 'Str. Def.', 'TD Acc.', 'TD Def.']
    )

    genders = resolve_fighter_gender(raw_fights_df, processed_fighters_df)
    ambiguous = processed_fighters_df[genders.isin(['andy', 'unknown'])]['Fighter Name']

    if len(ambiguous) > 0:
        print('The following fighters cannot have their gender specified, please complete it yourself, in the generated csv file')
        print('\n'.join(list(ambiguous)))
        print('\nYou should fill their genders in the csv or in the dictionary only in the case you want to use them to predict a fight for them.')

    processed_fighters_df.insert(1, 'Gender', genders)

    processed_fighters_df['Height'] = processed_fighters_df['Height'].apply(lambda x: convert_feet_and_inches_to_cm(x))
    processed_fighters_df['Weight'] = processed_fighters_df['Weight'].apply(lambda x: convert_weight_to_int(x))
    processed_fighters_df['Reach']  = processed_fighters_df['Reach'].apply(lambda x: convert_inches_to_cm(x))
    processed_fighters_df['Stance'] = processed_fighters_df['Stance'].apply(lambda x: 'Unknown' if pd.isna(x) else x)

    processed_fighters_df['DOB'] = pd.to_datetime(processed_fighters_df['DOB'])

    return processed_fighters_df


def main():
    raw_fights_df = retrieve_df_from_csv(RAW_FIGHT_STATS_FILENAME)
    raw_fighters_df = retrieve_df_from_csv(RAW_FIGHTER_STATS_FILENAME)

    processed_fighters_df = process_raw_fighters_df(raw_fighters_df, raw_fights_df)

    write_resulting_csv(
        processed_fighters_df,
        PROCESSED_FIGHTER_STATS_FILENAME
    )

    return 0

if __name__ == "__main__":
    main()
