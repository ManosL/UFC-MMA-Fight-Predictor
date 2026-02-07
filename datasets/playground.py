import pandas as pd


fighter_df_columns = ['Fighter_ID', 'Fighter_Name', 'DOB',
                      'Wins', 'Loses', 'Draws']

fighter_df = [
    [1, 'Khabib',   'Sep 20, 1988', 29, 1, 0],
    [2, 'McGregor', 'Jul 14, 1988', 22, 6, 0]
]

fights_df_columns = ['Fight_ID', 'Fight_Date', 'Result', 'Duration_Mins', 'Fighter_1_ID']

fights_df = [
    [1, 'Jan 01, 2010', 'win', 15, 1],
    [2, 'Feb 15, 2010', 'win', 3, 2],
    [3, 'Oct 15, 2010', 'lose', 13, 2],
    [4, 'Feb 08, 2011', 'win', 4.33, 1],
    [5, 'Feb 08, 2011', 'win', 14, 2],
    [6, 'Aug 15, 2011', 'lose', 25, 1],
    [7, 'Nov 17, 2011', 'win', 2, 1],
    [8, 'Feb 08, 2012', 'win', 0.3, 2],
    [9, 'Dec 08, 2012', 'win', 1, 2]
]

fighter_df = pd.DataFrame(fighter_df, columns=fighter_df_columns)
fights_df  = pd.DataFrame(fights_df, columns=fights_df_columns)

# Average Time in Mins
# fights_df['Shifted'] = fights_df.groupby('Fighter_1_ID')['Duration_Mins'].shift(periods=1)

fights_df_grouped = fights_df.groupby('Fighter_1_ID')

print(fights_df_grouped['Duration_Mins'].transform(lambda x: x.shift().cumsum()))
fights_df_avg_time = fights_df_grouped['Duration_Mins'].transform(lambda x: x.shift().cumsum()) / \
                         fights_df_grouped['Duration_Mins'].cumcount()

fights_df_avg_time = fights_df_avg_time.fillna(0.0)

print(fights_df_avg_time)

# Get record at time of fight
fights_df['is_win']  = (fights_df['Result'] == 'win').astype(int)
fights_df['is_lose'] = (fights_df['Result'] == 'lose').astype(int)
fights_df['is_draw'] = (fights_df['Result'] == 'draw').astype(int)

joined_df = fights_df.merge(fighter_df, how='inner', left_on='Fighter_1_ID',
                            right_on='Fighter_ID')

joined_df = joined_df[['Fight_ID', 'Fighter_ID', 'is_win', 'is_lose', 'is_draw',
                       'Wins', 'Loses', 'Draws']]
assert len(joined_df) == len(fights_df)

print(joined_df)

grouped_df = joined_df.groupby('Fighter_ID')

ufc_record = grouped_df[['is_win', 'is_lose', 'is_draw']].transform(lambda x: x.sum())

until_fight_ufc_record = grouped_df[['is_win', 'is_lose', 'is_draw']].transform(lambda x: x.shift().cumsum())
until_fight_ufc_record = until_fight_ufc_record.fillna(0)

resulting_df = pd.DataFrame()
resulting_df[['Wins', 'Loses', 'Draws']] = joined_df[['Wins', 'Loses', 'Draws']].to_numpy() - \
                                            (ufc_record - until_fight_ufc_record).to_numpy()
resulting_df = resulting_df.astype(int)

print((ufc_record - until_fight_ufc_record).to_numpy())
print(resulting_df)