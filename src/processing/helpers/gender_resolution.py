import pandas as pd

from constants import (
    FIRST_UFC_FEMALE_FIGHT_DATE
)

from fighter_unknown_gender_map import full_names_to_gender
import gender_guesser.detector as gender

COMMON_FIGHTER_ID_COL_NAME = "fighter_id"
UNKNOWN_GENDER_VALUE = "unknown"


def resolve_fight_gender(
    df: pd.DataFrame,
    gender_col_name: str = "Gender",
    fighter_1_id_col_name: str = "Fighter 1 ID",
    fighter_2_id_col_name: str = "Fighter 2 ID"
) -> pd.Series:
    new_genders = df[gender_col_name].copy()

    # Women first fought in UFC before 2013-02-23, thus, with this info we can
    # derive some unknown genders
    mask = (df['Fight Date'] < FIRST_UFC_FEMALE_FIGHT_DATE) & \
            (new_genders == UNKNOWN_GENDER_VALUE)

    new_genders.loc[mask] = 'male'

    # Fix "unknown" gender due to catch weight bouts by checking the gender from another fight
    known = df.loc[new_genders != UNKNOWN_GENDER_VALUE].copy()
    known[gender_col_name] = new_genders.loc[known.index]

    id_gender_1 = known[[fighter_1_id_col_name, gender_col_name]].rename(
        columns={
            fighter_1_id_col_name: COMMON_FIGHTER_ID_COL_NAME
        }
    )

    id_gender_2 = known[[fighter_2_id_col_name, gender_col_name]].rename(
        columns={
            fighter_2_id_col_name: COMMON_FIGHTER_ID_COL_NAME
        }
    )

    id_gender = pd.concat([id_gender_1, id_gender_2], ignore_index=True)

    # choose the most common known gender per fighter_id
    gender_map = id_gender \
        .groupby(COMMON_FIGHTER_ID_COL_NAME)[gender_col_name] \
        .agg(lambda x: x.mode().iloc[0])

    fill_1 = df[fighter_1_id_col_name].map(gender_map)
    fill_2 = df[fighter_2_id_col_name].map(gender_map)

    new_genders = new_genders.mask(
        new_genders.eq(UNKNOWN_GENDER_VALUE),
        fill_1.combine_first(fill_2)
    ).fillna(UNKNOWN_GENDER_VALUE)

    return new_genders


def resolve_gender_from_detector(
    detector: gender.Detector, 
    name: str
) -> str:
    fighters_gender = detector.get_gender(name.split()[0])

    if fighters_gender in {"male", 'mostly_male'}:
        return "male"
    
    if fighters_gender in {"female", 'mostly_female'}:
        return "female"
    
    return UNKNOWN_GENDER_VALUE


def determine_gender(
    fights: pd.DataFrame,
    gender_detector: gender.Detector,
    gender_col_name: str,
    fighter_name_col_name: str
) -> str:
    fighter_name = fights[fighter_name_col_name].iloc[0]
    fighter_does_not_have_fights = fights["Fight Date"].isna().all()

    gender = UNKNOWN_GENDER_VALUE

    if fighter_does_not_have_fights:
        gender = resolve_gender_from_detector(gender_detector, fighter_name)
    else:
        min_fight_date = fights["Fight Date"].min()

        if min_fight_date < FIRST_UFC_FEMALE_FIGHT_DATE:
            gender = "male"
        else:
            known_gender_fights = fights.loc[
                fights[gender_col_name] != UNKNOWN_GENDER_VALUE,
                gender_col_name
            ]

            if not known_gender_fights.empty:
                gender = known_gender_fights.mode().iloc[0]

    if gender == UNKNOWN_GENDER_VALUE:
        if fighter_name in full_names_to_gender.keys():
            gender = full_names_to_gender[fighter_name]

    return gender


def resolve_fighter_gender(
    fights_df: pd.DataFrame,
    fighters_df: pd.DataFrame,
    gender_col_name: str = "Gender",
    fighter_id_col_name: str = "Fighter ID",
    fighter_name_col_name: str = "Fighter Name",
    fighter_1_id_col_name: str = "Fighter 1 ID",
    fighter_2_id_col_name: str = "Fighter 2 ID"
) -> pd.Series:
    detector = gender.Detector()

    fights_df_gender = fights_df[["Fight Date", fighter_1_id_col_name, fighter_2_id_col_name, gender_col_name]].copy()
    fights_df_gender = fights_df_gender.melt(
        id_vars=["Fight Date", gender_col_name],
        value_vars=[fighter_1_id_col_name, fighter_2_id_col_name],
        var_name="fighter_position",
        value_name=fighter_id_col_name
    )
    
    fighters_df_copy = fighters_df[[fighter_id_col_name, fighter_name_col_name]].copy()

    total_appearances = fighters_df_copy.merge(
        fights_df_gender, 
        how="left", 
        on=fighter_id_col_name
    )

    total_appearances_grouped = total_appearances.groupby(by=fighter_id_col_name)
    gender_map = total_appearances_grouped.apply(
        lambda group: determine_gender(
            group,
            detector,
            gender_col_name,
            fighter_name_col_name
        ),
        include_groups=False
    )

    genders = fighters_df[fighter_id_col_name].map(gender_map).fillna(UNKNOWN_GENDER_VALUE)

    return genders
