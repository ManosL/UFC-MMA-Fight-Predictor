import pandas as pd

from constants import (
    RAW_FIGHT_STATS_FILENAME,
    PROCESSED_FIGHT_STATS_FILENAME
)

from helpers.gender_resolution import (
    resolve_fight_gender
)

from helpers.io import (
    retrieve_df_from_csv,
    write_resulting_csv
)
from helpers.transform import (
    normalize_no_val_to_nan,
    convert_end_time_in_total_fight_mins
)

from helpers.filtering import (
    filter_fights_with_valid_time_formats
)


def clean_and_process_raw_fights_df(raw_fights_df: pd.DataFrame) -> pd.DataFrame:
    processed_fights_df = raw_fights_df.copy()
    processed_fights_df['Fight Date'] = pd.to_datetime(processed_fights_df['Fight Date'])

    processed_fights_df["Gender"] = resolve_fight_gender(processed_fights_df)

    # I will only keep the matches that are 3 or 5 rounds of 5 minutes because fighters
    # of other fight formats had probably retired and because the requirements to win
    # might be different because at that times the rules were another for example.

    processed_fights_df = filter_fights_with_valid_time_formats(
        processed_fights_df,
        valid_fight_time_formats=[
            '3Rnd(5-5-5)',
            '5Rnd(5-5-5-5-5)',
            '3Rnd+OT(5-5-5-5)'
        ]
    )

    processed_fights_df = processed_fights_df.sort_values(by='Fight Date')

    processed_fights_df['Duration_Mins'] = processed_fights_df.apply(
        lambda x: convert_end_time_in_total_fight_mins(x),
        axis = 1
    )

    processed_fights_df = normalize_no_val_to_nan(
        processed_fights_df,
        nan_repr=["No Stats", "--", "---"]
    )

    return processed_fights_df


def main():
    raw_fights_df = retrieve_df_from_csv(RAW_FIGHT_STATS_FILENAME)
    processed_fights_df = clean_and_process_raw_fights_df(raw_fights_df)

    write_resulting_csv(
        processed_fights_df,
        PROCESSED_FIGHT_STATS_FILENAME
    )
    return 0


if __name__ == '__main__':
    main()
