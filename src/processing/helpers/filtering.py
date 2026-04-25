import pandas as pd


def filter_fights_with_valid_time_formats(
    df: pd.DataFrame,
    valid_fight_time_formats: list[str],
    fight_time_format_col: str = "Fight Time Format"
) -> pd.DataFrame:
    filtered_df = df[df[fight_time_format_col].isin(valid_fight_time_formats)]

    return filtered_df
