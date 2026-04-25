import re
import numpy as np
import pandas as pd


def normalize_no_val_to_nan(
    df: pd.DataFrame,
    nan_repr: list[str]
) -> pd.DataFrame:
    df = df.replace(nan_repr, np.nan)
    return df


def convert_percentage_features_to_decimals(
    df: pd.DataFrame,
    percentage_features: list[str]
) -> pd.DataFrame:
    for feature in percentage_features:
        df[feature] = df[feature].apply(
            lambda x: float(x.strip('%')) / 100 if x is not np.nan
            else np.nan
        )

    return df


# Should be an str of form mm:ss
def convert_time_str_to_mins(elem: str) -> float:
    mins = int(elem.split(':')[0])
    secs = int(elem.split(':')[1])

    return_val = (mins * 60 + secs) / 60

    return return_val


def convert_feet_and_inches_to_cm(x: str) -> float:
    if pd.isna(x):
        return np.nan

    res = re.match(r"(\d+)\'\s+(\d+)\"", x)

    if not res or not res.group(1) or not res.group(2):
        raise ValueError(f"Could not parse \"{x}\" to extract feet and inches to convert to cm")

    feet = int(res.group(1))
    inches = int(res.group(2))

    return (feet * 12 + inches) * 2.54


def convert_weight_to_int(x: str) -> int:
    if pd.isna(x):
        return np.nan

    res = re.match(r"(\d+)\s+lbs\.", x)

    if not res or not res.group(1):
        raise ValueError(f"Could not parse \"{x}\" to extract fighter's weight")

    return int(res.group(1))


def convert_inches_to_cm(x: str) -> float:
    if pd.isna(x) or len(x) == 0:
        return np.nan

    res = re.match(r"(\d+)\"", x)

    if not res or not res.group(1):
        raise ValueError(f"Could not parse \"{x}\" to extract inches to convert to cm")

    inches = int(res.group(1))

    return (inches) * 2.54


def convert_end_time_in_total_fight_mins(
    row: pd.Series,
    end_round_col: str = "Round",
    end_time_col: str = "Time"
) -> float:
    fight_time =  (int(row[end_round_col]) - 1) * 5.0

    end_round_mins = int(row[end_time_col].split(':')[0])
    end_round_secs = int(row[end_time_col].split(':')[1])

    fight_time += (end_round_mins +  end_round_secs / 60.0)
    return fight_time
