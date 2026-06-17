import pandas as pd


FIRST_UFC_FEMALE_FIGHT_DATE      = pd.to_datetime("2013-02-23")
RAW_FIGHT_STATS_FILENAME         = "fight_new_actual_stats.csv"
RAW_FIGHTER_STATS_FILENAME       = "fighters_new_current_stats.csv"
PROCESSED_FIGHT_STATS_FILENAME   = "fight_new_actual_stats_processed.csv"
PROCESSED_FIGHTER_STATS_FILENAME = "fighters_new_current_stats_processed.csv"

FIGHT_INTEGER_COLUMNS = [
    'Knock Downs', 'Sign.Strikes Done',
    'Sign.Strikes Attempted', 'Total Strikes Done',
    'Total Strikes Attempted', 'Takedowns Done', 'Takedowns Attempted',
    'Submission Attempts', 'Rev.'
]

def get_fight_df_integer_columns():
    fighter_1_prefix = 'Fighter 1 '
    fighter_2_prefix = 'Fighter 2 '
    
    fighter_1_columns = [f'{fighter_1_prefix}{col}' for col in FIGHT_INTEGER_COLUMNS]
    fighter_2_columns = [f'{fighter_2_prefix}{col}' for col in FIGHT_INTEGER_COLUMNS]

    return fighter_1_columns + fighter_2_columns