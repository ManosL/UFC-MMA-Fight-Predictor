import pandas as pd



# 'Age' == FightDate - Fighters DOB in years
def get_age_at_time_of_fight(fighters_df, fighters_fights_df):
    joined_df = fighters_fights_df.merge(fighters_df, how='left',
                                         left_on='Fighter 1 ID',
                                         right_on='Fighter ID')

    joined_df['Fight Date'] = pd.to_datetime(joined_df['Fight Date'])
    joined_df['DOB'] = pd.to_datetime(joined_df['DOB'])

    return ((joined_df['Fight Date'] - joined_df['DOB']).dt.days // 365.25).fillna(0).astype(int)



def get_record_at_time_of_fight(fighters_df, fighters_fights_df):
    joined_df = fighters_fights_df.merge(fighters_df, how='inner',
                                         left_on='Fighter 1 ID',
                                         right_on='Fighter ID')

    print(joined_df.columns)
    joined_df = joined_df[['Fight_ID', 'Fight Date', 'Fighter ID', 'Result',
                            'Fighter Name', 'Wins', 'Loses', 'Draws']]

    # Convert the result to one-hot encoding
    joined_df['is_win']  = (joined_df['Result'] == 'win').astype(int)
    joined_df['is_lose'] = (joined_df['Result'] == 'lose').astype(int)
    joined_df['is_draw'] = (joined_df['Result'] == 'draw').astype(int)

    assert len(joined_df) == len(fighters_fights_df)

    grouped_df = joined_df.groupby('Fighter ID')

    # Getting each fighter's UFC record. Doing transform in order to work as
    # a window function and each row have that record in order to do the
    # calculations easier
    ufc_record = grouped_df[['is_win', 'is_lose', 'is_draw']].transform('sum')

    # Applying cumsum with shift in order to get the UFC record of the fighter
    # at the time of fight
    until_fight_ufc_record = grouped_df[['is_win', 'is_lose', 'is_draw']].transform(lambda x: x.shift().cumsum())
    until_fight_ufc_record = until_fight_ufc_record.fillna(0)

    # Doing Total Record - (Total UFC record - Current UFC Record) in order to get
    # full record at time of the fight.
    resulting_df = pd.DataFrame()
    resulting_df[['Wins', 'Loses', 'Draws']] = joined_df[['Wins', 'Loses', 'Draws']].to_numpy() - \
                                                (ufc_record - until_fight_ufc_record).to_numpy()
    resulting_df = resulting_df.astype(int)

    return resulting_df



# 'Avg_Time(MINS)' == Avg Time in minutes up until the present fight
def get_avg_fight_time(fighters_fights_df):
    fighter_fights_df_grouped = fighters_fights_df.groupby('Fighter 1 ID')

    fighter_fights_df_avg_time = \
        fighter_fights_df_grouped['Duration_Mins'].transform(lambda x: x.shift().cumsum()) / \
        fighter_fights_df_grouped['Duration_Mins'].cumcount()

    fighter_fights_df_avg_time = fighter_fights_df_avg_time.fillna(0.0)

    return fighter_fights_df_avg_time



def get_fighter_height(fighters_df, fighters_fights_df):
    joined_df = fighters_fights_df.merge(fighters_df, how='left',
                                         left_on='Fighter 1 ID',
                                         right_on='Fighter ID')

    return joined_df['Height']



def get_fighter_reach(fighters_df, fighters_fights_df):
    joined_df = fighters_fights_df.merge(fighters_df, how='left',
                                         left_on='Fighter 1 ID',
                                         right_on='Fighter ID')

    return joined_df['Reach']



def get_fighter_stance(fighters_df, fighters_fights_df):
    joined_df = fighters_fights_df.merge(fighters_df, how='left',
                                         left_on='Fighter 1 ID',
                                         right_on='Fighter ID')

    return joined_df['Stance']



# TODO: BECAUSE THE BELOW FUNCTIONS LOOK VERY SIMILAR LOOK HOW TO NOT REPEAT YOURSELF

# 'Sign_SLpMin' == SUM(Sign Strikes Landed up until the fight) / SUM(Total Mins up until the fight)
def get_sign_strikes_landed_per_min(fighters_fights_df):
    fighter_fights_df_grouped = fighters_fights_df.groupby('Fighter 1 ID')

    fighter_fights_df_sslpm = \
        fighter_fights_df_grouped['Fighter 1 Sign.Strikes Done'].transform(lambda x: x.astype(int).shift().cumsum()) / \
        fighter_fights_df_grouped['Duration_Mins'].transform(lambda x: x.shift().cumsum())

    fighter_fights_df_sslpm = fighter_fights_df_sslpm.fillna(0.0)

    print(fighter_fights_df_sslpm[fighters_fights_df['Fighter 1 ID'] == 'f4c49976c75c5ab2'])
    return fighter_fights_df_sslpm



# 'Str_Acc'(percentage) == SUM(Sign Strikes Landed up until the fight) / SUM(Sign Strikes Attempted up until the fight)
def get_striking_accuracy(fighters_fights_df):
    fighter_fights_df_grouped = fighters_fights_df.groupby('Fighter 1 ID')

    fighter_fights_df_str_acc = \
        fighter_fights_df_grouped['Fighter 1 Sign.Strikes Done'].transform(lambda x: x.astype(int).shift().cumsum()) / \
        fighter_fights_df_grouped['Fighter 1 Sign.Strikes Attempted'].transform(lambda x: x.astype(int).shift().cumsum())

    fighter_fights_df_str_acc = fighter_fights_df_str_acc * 100.0

    fighter_fights_df_str_acc = fighter_fights_df_str_acc.fillna(0.0)

    return fighter_fights_df_str_acc



# 'Sign_SApMin' == SUM(Opponent Sign Strikes Landed up until the fight) / SUM(Total Mins up until the fight)
def get_sign_strikes_absorbed_per_min(fighters_fights_df):
    fighter_fights_df_grouped = fighters_fights_df.groupby('Fighter 1 ID')

    fighter_fights_df_ssapm = \
        fighter_fights_df_grouped['Fighter 2 Sign.Strikes Done'].transform(lambda x: x.astype(int).shift().cumsum()) / \
        fighter_fights_df_grouped['Duration_Mins'].transform(lambda x: x.shift().cumsum())

    fighter_fights_df_ssapm = fighter_fights_df_ssapm.fillna(0.0)

    return fighter_fights_df_ssapm



#'Defense'(percentage) == SUM(Opponent Sign Strikes NOT Landed up until the fight) / SUM(Opponent Sign Strikes
#                                                                            Attempted up until the fight)
def get_defense(fighters_fights_df):
    fighter_fights_df_grouped = fighters_fights_df.groupby('Fighter 1 ID')

    fighter_fights_df_defense = \
        fighter_fights_df_grouped['Fighter 2 Sign.Strikes Done'].transform(lambda x: x.astype(int).shift().cumsum()) / \
        fighter_fights_df_grouped['Fighter 2 Sign.Strikes Attempted'].transform(lambda x: x.astype(int).shift().cumsum())

    fighter_fights_df_defense = (1.0 - fighter_fights_df_defense) * 100

    fighter_fights_df_defense = fighter_fights_df_defense.fillna(0.0)

    return fighter_fights_df_defense



# 'Takedown_Avgp15M' == SUM(Takedown Landed up until the fight) * 15 / SUM(Total Mins up until the fight)
def get_takedowns_landed_per_15_mins(fighters_fights_df):
    fighter_fights_df_grouped = fighters_fights_df.groupby('Fighter 1 ID')

    fighter_fights_df_tdlp15m = \
        fighter_fights_df_grouped['Fighter 1 Takedowns Done'].transform(lambda x: x.astype(int).shift().cumsum()) / \
        fighter_fights_df_grouped['Duration_Mins'].transform(lambda x: x.shift().cumsum() / 15.0)

    fighter_fights_df_tdlp15m = fighter_fights_df_tdlp15m.fillna(0.0)

    return fighter_fights_df_tdlp15m



# 'Takedown_Acc' == SUM(Takedowns Landed up until the fight) / SUM(Takedowns Attempted up until the fight)
def get_takedown_accuracy(fighters_fights_df):
    fighter_fights_df_grouped = fighters_fights_df.groupby('Fighter 1 ID')

    fighter_fights_df_td_acc = \
        fighter_fights_df_grouped['Fighter 1 Takedowns Done'].transform(lambda x: x.astype(int).shift().cumsum()) / \
        fighter_fights_df_grouped['Fighter 1 Takedowns Attempted'].transform(lambda x: x.astype(int).shift().cumsum())

    fighter_fights_df_td_acc = fighter_fights_df_td_acc * 100.0

    fighter_fights_df_td_acc = fighter_fights_df_td_acc.fillna(0.0)

    return fighter_fights_df_td_acc



# 'Takedown_Def' == SUM(Opponent Takedowns NOT Landed up until the fight) / SUM(Opponent Takedowns
#                                                                            Attempted up until the fight)
def get_takedown_defense(fighters_fights_df):
    fighter_fights_df_grouped = fighters_fights_df.groupby('Fighter 1 ID')

    fighter_fights_df_td_defense = \
        fighter_fights_df_grouped['Fighter 2 Takedowns Done'].transform(lambda x: x.astype(int).shift().cumsum()) / \
        fighter_fights_df_grouped['Fighter 2 Takedowns Attempted'].transform(lambda x: x.astype(int).shift().cumsum())

    fighter_fights_df_td_defense = (1.0 - fighter_fights_df_td_defense) * 100

    fighter_fights_df_td_defense = fighter_fights_df_td_defense.fillna(0.0)

    return fighter_fights_df_td_defense



# 'Sub_Avgp15M' == SUM(Submission Attempts until the fight) * 15 / SUM(Total Mins up until the fight)
def get_submission_attempts_per_15_mins(fighters_fights_df):
    fighter_fights_df_grouped = fighters_fights_df.groupby('Fighter 1 ID')

    fighter_fights_df_smap15m = \
        fighter_fights_df_grouped['Fighter 1 Submission Attempts'].transform(lambda x: x.astype(int).shift().cumsum()) / \
        fighter_fights_df_grouped['Duration_Mins'].transform(lambda x: x.shift().cumsum() / 15.0)

    fighter_fights_df_smap15m = fighter_fights_df_smap15m.fillna(0.0)

    return fighter_fights_df_smap15m
