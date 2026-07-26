FIGHTER_1_PREFIX = "Fighter_1_"
FIGHTER_2_PREFIX = "Fighter_2_"
DIFFERENCE_FEATURES_SUFFIX = "_Difference"

FIGHT_ID_COLUMN = "Fight_ID"
FIGHT_DATE_COLUMN = "Fight_Date"
COMMON_CATEGORICAL_FEATURES = [
    "Gender", "Weight_Class", "Title_Fight","Fight_Time_Format"
]

PER_FIGHTER_ID_COLUMNS = ["ID", "Name"]
PER_FIGHTER_NUMERIC_COLUMNS = [
    "Age", "Wins", "Loses", "Draws", "Avg_Time(MINS)", "Height",
    "Reach", "Sign_SLpMin", "Str_Acc", "Sign_SApMin", "Defense",
    "Takedown_Avgp15M", "Takedown_Acc", "Takedown_Def", "Sub_Avgp15M"
]
PER_FIGHTER_CATEGORICAL_FEATURES = ["Stance"]

LABEL_COLUMNS = ['Result', 'Method', 'Round', 'Time']
