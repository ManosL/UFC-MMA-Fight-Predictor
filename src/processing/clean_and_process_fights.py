import os
import argparse
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


def main(version_id: str) -> int:
    raw_fights_df = retrieve_df_from_csv(
        os.path.join(version_id, RAW_FIGHT_STATS_FILENAME)
    )
    processed_fights_df = clean_and_process_raw_fights_df(raw_fights_df)

    write_resulting_csv(
        processed_fights_df,
        os.path.join(version_id, PROCESSED_FIGHT_STATS_FILENAME)
    )
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--version_id", "-v", help="Version ID")
    parser.parse_args()
    main(parser.version_id)
