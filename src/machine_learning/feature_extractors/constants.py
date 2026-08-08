FIGHTER_1_PREFIX = "Fighter_1_"
FIGHTER_2_PREFIX = "Fighter_2_"
DIFFERENCE_FEATURES_SUFFIX = "_Difference"

FIGHT_ID_COLUMN = "Fight_ID"
FIGHT_DATE_COLUMN = "Fight_Date"
COMMON_CATEGORICAL_FEATURES = [
    "Gender", "Weight_Class", "Title_Fight", "Fight_Time_Format",
]
COMMON_PERCENTAGE_FEATURES = []

PER_FIGHTER_ID_COLUMNS = ["ID", "Name"]
PER_FIGHTER_NUMERIC_COLUMNS = [
    "Age", "Wins", "Loses", "Draws", "Avg_Time(MINS)", "Height",
    "Reach", "Sign_SLpMin", "Str_Acc", "Sign_SApMin", "Defense",
    "Takedown_Avgp15M", "Takedown_Acc", "Takedown_Def", "Sub_Avgp15M",
]
PER_FIGHTER_PERCENTAGE_FEATURES = [
    "Str_Acc",
    "Defense",
    "Takedown_Acc",
    "Takedown_Def",
]

PER_FIGHTER_CATEGORICAL_FEATURES = ["Stance"]

LABEL_COLUMNS = ['Result', 'Method', 'Round', 'Time']

GENDER_MAP = {
    "male": 1,
    "female": 0,
}

TITLE_FIGHT_MAP = {
    True: 1,
    False: 0,
}

WEIGHT_CLASS_MAP = {
    'catch weight': 9,
    'heavyweight': 8,
    'light heavyweight': 7,
    'middleweight': 6,
    'welterweight': 5,
    'lightweight': 4,
    'featherweight': 3,
    'bantamweight': 2,
    'flyweight': 1,
    'strawweight': 0,
}

FIGHT_TIME_FORMAT_MAP = {
    '3rnd(5-5-5)': 0,
    '3rnd+ot(5-5-5-5)': 0,
    '5rnd(5-5-5-5-5)': 1,
}

STANCE_MAP = {
    'orthodox': 0,
    'southpaw': 1,
    'open stance': 2,
    'switch': 3,
    'sideways': 4,
}
