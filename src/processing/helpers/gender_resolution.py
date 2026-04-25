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


def resolve_fighter_gender(
    fights_df: pd.DataFrame,
    fighters_df: pd.DataFrame
) -> pd.Series:
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
                else:
                    fighters_gender = 'unknown'

            genders.append(fighters_gender)
        else:
            mask = (fights_df['Fighter 1 ID'] == fighter_id) | (fights_df['Fighter 2 ID'] == fighter_id)
            min_fight_date = fights_df[mask]["Fight Date"].min()

            if min_fight_date < FIRST_UFC_FEMALE_FIGHT_DATE:
                fighters_gender = 'male'
            else:
                fighters_gender = fighters_fights_1[fighters_fights_1['Gender'] != 'unknown']['Gender'].mode()

                if fighters_gender.empty:
                    fighters_gender = fighters_fights_2[fighters_fights_2['Gender'] != 'unknown']['Gender'].mode()

                fighters_gender = 'unknown' if fighters_gender.empty else fighters_gender.iloc[0]

                if fighters_gender == 'unknown':
                    if fighters_names[i] in full_names_to_gender.keys():
                        fighters_gender = full_names_to_gender[fighters_names[i]]

            genders.append(fighters_gender)

    return pd.Series(genders)
