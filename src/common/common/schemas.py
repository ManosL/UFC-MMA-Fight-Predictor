CRAWLED_FIGHTER_COLUMNS = [
    'Fighter ID', 'Fighter Name', 'Wins', 'Loses', 'Draws', 'Height',
    'Weight', 'Reach', 'Stance', 'DOB', 'SLpM', 'Str.Acc.', 'SApM',
    'Str. Def.', 'TD Avg.', 'TD Acc.', 'TD Def.', 'Sub. Avg.'
]

__CRAWLED_FIGHT_COMMON_COLUMNS = [
    'Fight_ID', 'Fight Date', 'Gender', 
    'Weight Class', 'Title Fight', 'Result', 
    'Method', 'Round', 'Time', 'Fight Time Format'
]

__CRAWLED_FIGHT_FIGHTER_COLUMNS = [
    'ID', 'Name', 'Nickname', 'Knock Downs', 'Sign.Strikes Done',
    'Sign.Strikes Attempted', 'Sign.Strikes Perc.', 'Total Strikes Done',
    'Total Strikes Attempted', 'Takedowns Done', 'Takedowns Attempted',
    'Takedowns Perc.', 'Submission Attempts', 'Rev.', 'Control Time'
]

def get_crawled_fight_columns():
    fighter_1_prefix = 'Fighter 1 '
    fighter_2_prefix = 'Fighter 2 '
    
    fighter_1_columns = [f'{fighter_1_prefix}{col}' for col in __CRAWLED_FIGHT_FIGHTER_COLUMNS]
    fighter_2_columns = [f'{fighter_2_prefix}{col}' for col in __CRAWLED_FIGHT_FIGHTER_COLUMNS]

    return __CRAWLED_FIGHT_COMMON_COLUMNS + fighter_1_columns + fighter_2_columns
