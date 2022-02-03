import pandas as pd
import numpy  as np
from math import isnan

def retrieve_initial_dfs(event_df_path, fighters_df_path):
    init_event_df = pd.read_csv(event_df_path, sep = '|', header = None,
    names = ['Fight Date','Gender','Weight Class','Title Fight','Result','Method','Round','Time','Fight Time Format',
        'Fighter 1 ID', 'Fighter 1 Name', 'Fighter 1 Nickname',
		'Fighter 1 Knock Downs', 'Fighter 1 Sign.Strikes Done','Fighter 1 Sign.Strikes Attempted',
		'Fighter 1 Sign.Strikes Perc.','Fighter 1 Total Strikes Done',
		'Fighter 1 Total Strikes Attempted',
		'Fighter 1 Takedowns Done', 'Fighter 1 Takedowns Attempted', 'Fighter 1 Takedowns Perc.',
		'Fighter 1 Submission Attempts', 'Fighter 1 Rev', 'Fighter 1 Control',
        'Fighter 2 ID', 'Fighter 2 Name', 'Fighter 2 Nickname',
		'Fighter 2 Knock Downs', 'Fighter 2 Sign.Strikes Done','Fighter 2 Sign.Strikes Attempted',
		'Fighter 2 Sign.Strikes Perc.','Fighter 2 Total Strikes Done',
		'Fighter 2 Total Strikes Attempted',
		'Fighter 2 Takedowns Done', 'Fighter 2 Takedowns Attempted', 'Fighter 2 Takedowns Perc.',
		'Fighter 2 Submission Attempts', 'Fighter 2 Rev', 'Fighter 2 Control'])

    fighters_df = pd.read_csv(fighters_df_path, sep='|', header=0)

    return init_event_df, fighters_df

def preprocess_initial_dfs(init_event_df, fighters_df):
    init_event_df['Fight Date'] = pd.to_datetime(init_event_df['Fight Date'])

    # I will only keep the matches that are 3 or 5 rounds of 5 minutes because fighters
    # of other fight formats had probably retired and because the requirements to win
    # might be different because at that times the rules were another for example.

    init_event_df = init_event_df[init_event_df['Fight Time Format'].isin(['3Rnd(5-5-5)', '5Rnd(5-5-5-5-5)'])]    
    init_event_df = init_event_df.sort_values(by='Fight Date')

    f1_total_strikes_done = init_event_df['Fighter 1 Total Strikes Done']
    f1_total_strikes_attempted = init_event_df['Fighter 1 Total Strikes Attempted']

    # This is if I want to add 2 extra features
    f1_total_strikes_perc = []
    for stat in zip(f1_total_strikes_done,f1_total_strikes_attempted):
        if stat[0] == 'No Stats' or stat[1] == 'No Stats':
            f1_total_strikes_perc.append('No Stats')
        elif int(stat[1]) == 0:
            f1_total_strikes_perc.append(0)
        else:
            f1_total_strikes_perc.append(int((int(stat[0]) / int(stat[1])) * 100))

    f2_total_strikes_done = init_event_df['Fighter 2 Total Strikes Done']
    f2_total_strikes_attempted = init_event_df['Fighter 2 Total Strikes Attempted']

    f2_total_strikes_perc = []
    for stat in zip(f2_total_strikes_done,f2_total_strikes_attempted):
        if stat[0] == 'No Stats' or stat[1] == 'No Stats':
            f2_total_strikes_perc.append('No Stats')
        elif int(stat[1]) == 0:
            f2_total_strikes_perc.append(0)
        else:
            f2_total_strikes_perc.append(int((int(stat[0]) / int(stat[1])) * 100))
    
    ############################################

    init_event_df.insert(0, 'Fight_ID', range(1,len(init_event_df) + 1))
    #init_event_df.insert(17,'Fighter 1 Total Strikes Perc.',f1_total_strikes_perc)
    #init_event_df.insert(33,'Fighter 2 Total Strikes Perc.',f2_total_strikes_perc)

    init_event_df = init_event_df.replace('No Stats', np.nan)
    fighters_df = fighters_df.replace('No Stat', np.nan)

    return init_event_df, fighters_df

def create_final_df(init_event_df, fighter_fight_attrs, path_to_write):
    # fighter_fight_attrs is a dictionary of the form 
    # {Fighter_Name->{Fighter_ID->[list of Fighter_Name stats at time of fight with Fight_ID]}}
    
    # Now we need to join the fighter_fight_attrs table to extract the final dataset
    # This will be done by goings through each line of init_event_df and get the
    # the info from their corresponding fighter.
    resulting_event_df = []

    for _, row in init_event_df.iterrows():
        new_df_row = [row['Fight_ID'], row['Fight Date'], row['Gender'], row['Weight Class'],
                      row['Title Fight'], row['Result'], row['Method'], row['Round'], 
                      row['Time'], row['Fight Time Format']]
        
        row_fighter_1 = row['Fighter 1 ID']
        row_fighter_2 = row['Fighter 2 ID']

        fighter_1_attrs = fighter_fight_attrs[row_fighter_1][row['Fight_ID']][3:]
        fighter_2_attrs = fighter_fight_attrs[row_fighter_2][row['Fight_ID']][3:]

        final_fighter_1_attrs = [row['Fighter 1 ID'], row['Fighter 1 Name'], row['Fighter 1 Nickname']] + fighter_1_attrs
        final_fighter_2_attrs = [row['Fighter 2 ID'], row['Fighter 2 Name'], row['Fighter 2 Nickname']] + fighter_2_attrs

        new_df_row += final_fighter_1_attrs + final_fighter_2_attrs

        resulting_event_df.append(new_df_row)
    
    resulting_event_df_col_names = ['Fight_ID', 'Fight_Date', 'Gender', 'Weight_Class', 'Title_Fight', 'Result', 'Method', 'Round', 'Time',
                                'Fight_Time_Format', 'Fighter_1_ID', 'Fighter_1_Name', 'Fighter_1_Nickname', 'Fighter_1_Age', 'Fighter_1_Wins',
                                'Fighter_1_Loses', 'Fighter_1_Draws', 'Fighter_1_Avg_Time(MINS)', 'Fighter_1_Height', 
                                'Fighter_1_Reach', 'Fighter_1_Stance', 'Fighter_1_Sign_SLpMin', 'Fighter_1_Str_Acc',
                                'Fighter_1_Sign_SApMin', 'Fighter_1_Defense', 'Fighter_1_Takedown_Avgp15M', 'Fighter_1_Takedown_Acc', 
                                'Fighter_1_Takedown_Def', 'Fighter_1_Sub_Avgp15M', 'Fighter_2_ID', 'Fighter_2_Name', 'Fighter_2_Nickname', 
                                'Fighter_2_Age', 'Fighter_2_Wins', 'Fighter_2_Loses', 'Fighter_2_Draws', 'Fighter_2_Avg_Time(MINS)', 
                                'Fighter_2_Height', 'Fighter_2_Reach', 'Fighter_2_Stance', 'Fighter_2_Sign_SLpMin', 'Fighter_2_Str_Acc',
                                'Fighter_2_Sign_SApMin', 'Fighter_2_Defense', 'Fighter_2_Takedown_Avgp15M', 'Fighter_2_Takedown_Acc', 
                                'Fighter_2_Takedown_Def', 'Fighter_2_Sub_Avgp15M']
    
    resulting_event_df = pd.DataFrame(resulting_event_df, columns=resulting_event_df_col_names)

    resulting_event_df.to_csv(path_to_write, sep='|', na_rep='NaN', index=False)

    return

def main():
    init_event_datafile_path = './UFC_Fights_Train.tsv'
    init_fighters_datafile_path =  './UFC_Fighters.tsv'

    init_event_df, fighters_df = retrieve_initial_dfs(init_event_datafile_path, init_fighters_datafile_path)
    init_event_df, fighters_df = preprocess_initial_dfs(init_event_df, fighters_df)

    #print(init_event_df)

    fighters_fights_map = {}
    
    assert(len(fighters_df['Fighter ID']) == len(fighters_df))

    for fighter_id in fighters_df['Fighter ID'].unique():
        df1 = init_event_df[(init_event_df['Fighter 1 ID'] == fighter_id)]
        df2 = init_event_df[(init_event_df['Fighter 2 ID'] == fighter_id)]
 
        df1_tmp = pd.concat([df1['Fight_ID'], df1.iloc[:,10:25]],axis = 1)
        df1_opponent = pd.concat([df1['Fight_ID'], df1.iloc[:,25:40]],axis = 1)
        df1 = df1_tmp

        df2_tmp = pd.concat([df2['Fight_ID'], df2.iloc[:,25:40]],axis = 1)
        df2_opponent = pd.concat([df2['Fight_ID'], df2.iloc[:,10:25]],axis = 1)
        df2 = df2_tmp

        df1.columns = ['Fight_ID', 'Fighter ID', 'Fighter Name', 'Fighter Nickname',
		'Fighter Knock Downs', 'Fighter Sign.Strikes Done','Fighter Sign.Strikes Attempted',
		'Fighter Sign.Strikes Perc.','Fighter Total Strikes Done', 'Fighter Total Strikes Attempted',
		'Fighter Takedowns Done', 'Fighter Takedowns Attempted', 'Fighter Takedowns Perc.',
		'Fighter Submission Attempts', 'Fighter Rev', 'Fighter Control']

        df2.columns = df1.columns
        df1_opponent.columns = df1.columns
        df2_opponent.columns = df1.columns

        # Sorting each fighter's fights at ascendind date order
        # We create this dictionary to be able to access easily the fights of each fighter
        fighters_fights_map[fighter_id] = tuple([pd.concat([df1,df2], axis=0, ignore_index=True).sort_values(by='Fight_ID'),\
                                pd.concat([df1_opponent,df2_opponent], axis=0, ignore_index=True).sort_values(by='Fight_ID')])
    
    fighter_sums = {}
    fighter_fight_attrs = {}

    for fighter in fighters_fights_map.keys():
        # I didn't read any fighter line
        assert fighter not in fighter_fight_attrs.keys()

        # Initialize the attributes that can help me determine attributes that are
        # dependent from total time, total matches etc. 
        fighter_sums[fighter] = {'Wins': 0, 'Loses': 0, 'Draws': 0,'Matches': 0, 'Total_Time': 0, 
                                'Total_Sign_Str_Landed': 0, 'Total_Sign_Str_Attempt': 0,
                                'Total_Sign_Str_Absorbed': 0, 'Total_Sign_Str_Opp_Attempt': 0, 'Total_TD': 0,
                                'Total_TD_Attempt':0, 'Total_Opp_TD': 0, 'Total_Opp_TD_Attempt': 0,
                                'Total_Subs': 0}

        # Configuring the record of the fighter before his first match in the dataset
        #print(fighter)
        fighter_sums[fighter]['Wins'] = int(fighters_df[fighters_df['Fighter ID'] == fighter]['Wins'])
        fighter_sums[fighter]['Loses'] = int(fighters_df[fighters_df['Fighter ID'] == fighter]['Loses'])
        fighter_sums[fighter]['Draws'] = int(fighters_df[fighters_df['Fighter ID'] == fighter]['Draws'])
        
        for _, row in fighters_fights_map[fighter][0].iterrows():
            interest_row = init_event_df[init_event_df['Fight_ID'] == row['Fight_ID']]
            result = list(interest_row['Result'])[0]

            if result == 'draw':
                fighter_sums[fighter]['Draws'] -= 1
            elif result == 'win':
                if list(interest_row['Fighter 1 Name'])[0] == row['Fighter Name']:
                    fighter_sums[fighter]['Wins'] -= 1
                elif list(interest_row['Fighter 2 Name'])[0] == row['Fighter Name']:
                    fighter_sums[fighter]['Loses'] -= 1
                else:
                    print("I got wrong fighter name ",row['Fighter Name'])
                    print(row)
                    print(row['Fight_ID'])
            elif result == 'lose':
                if list(interest_row['Fighter 1 Name'])[0] == row['Fighter Name']:
                    fighter_sums[fighter]['Loses'] -= 1
                elif list(interest_row['Fighter 2 Name'])[0] == row['Fighter Name']:
                    fighter_sums[fighter]['Wins'] -= 1
                else:
                    print("I got wrong fighter name ",row['Fighter Name'])
                    print(row)
                    print(row['Fight_ID'])
            elif result == 'no contest':
                # do nothing
                fighter_sums[fighter]['Wins'] += 0
            else:
                print("I got wrong result ",result)
        
        #print(fighter)
        # I won't use this assertions, just if record < 0 then record = 0
        #assert fighter_sums[fighter]['Wins'] >= 0
        #assert fighter_sums[fighter]['Loses'] >= 0
        #assert fighter_sums[fighter]['Draws'] >= 0

        fighter_sums[fighter]['Wins']  = fighter_sums[fighter]['Wins']  if fighter_sums[fighter]['Wins']  >= 0 else 0
        fighter_sums[fighter]['Loses'] = fighter_sums[fighter]['Loses'] if fighter_sums[fighter]['Loses'] >= 0 else 0
        fighter_sums[fighter]['Draws'] = fighter_sums[fighter]['Draws'] if fighter_sums[fighter]['Draws'] >= 0 else 0

        fighter_fight_attrs[fighter] = {}   # Dict from Fight_ID to list of form ['Fight_ID', 'ID', 'Name','Age','Wins','Loses','Draws','Avg_Time(MINS)',
                                            #        'Height','Reach','Stance','Sign_SLpMin','Str_Acc','Sign_SApMin',
                                            #        'Defense','Takedown_Avgp15M','Takedown_Acc','Takedown_Def','Sub_Avgp15M'])

        for _, fighter_row in fighters_fights_map[fighter][0].iterrows():
            curr_fight_id = fighter_row['Fight_ID']
            
            total_fight_row = init_event_df[init_event_df['Fight_ID'] == curr_fight_id]
            opponent_row = fighters_fights_map[fighter][1][fighters_fights_map[fighter][1]['Fight_ID'] == curr_fight_id]
            
            fighter_df_row = fighters_df[fighters_df['Fighter Name'] == fighter_row['Fighter Name'].rstrip()]

            #print((fighter_df_row['DOB']))
            #print(fighter_row['Fighter Name'].rstrip())

            if list(fighter_df_row['DOB'].isnull())[0]:
                current_age = np.nan
            else:
                birth_year  = int(str(list(fighter_df_row['DOB'])[0]).split(' ')[2])
                fight_year  = int(str(list(total_fight_row['Fight Date'])[0]).split(' ')[0].split('-')[0]) # I will determine age with this way because is easier
                current_age = fight_year - birth_year

            fighter_attr_row = [curr_fight_id, fighter_row['Fighter ID'], fighter_row['Fighter Name'], current_age,\
                                fighter_sums[fighter]['Wins'], fighter_sums[fighter]['Loses'], fighter_sums[fighter]['Draws']]

            # If this is the first fight
            if fighter_sums[fighter]['Matches'] == 0:
                rounds = int(list(total_fight_row['Round'])[0])
                last_round_dur = list(total_fight_row['Time'])[0]
                last_round_dur = (int(last_round_dur.split(':')[0])*60 + int(last_round_dur.split(':')[1])) / 60
                fight_dur_mins = (rounds - 1) * 5 + last_round_dur # Because in all fights the round has 5 mins duration

                # Updating fighter's stats
                fighter_sums[fighter]['Total_Time'] += fight_dur_mins
                
                fighter_sums[fighter]['Total_Sign_Str_Landed']      += int(fighter_row['Fighter Sign.Strikes Done'])
                fighter_sums[fighter]['Total_Sign_Str_Attempt']     += int(fighter_row['Fighter Sign.Strikes Attempted'])
                fighter_sums[fighter]['Total_Sign_Str_Absorbed']    += int(list(opponent_row['Fighter Sign.Strikes Done'])[0])
                fighter_sums[fighter]['Total_Sign_Str_Opp_Attempt'] += int(list(opponent_row['Fighter Sign.Strikes Attempted'])[0])
                fighter_sums[fighter]['Total_TD']                   += int(fighter_row['Fighter Takedowns Done'])
                fighter_sums[fighter]['Total_TD_Attempt']           += int(fighter_row['Fighter Takedowns Attempted'])
                fighter_sums[fighter]['Total_Opp_TD']               += int(opponent_row['Fighter Takedowns Done'])
                fighter_sums[fighter]['Total_Opp_TD_Attempt']       += int(opponent_row['Fighter Takedowns Attempted'])
                fighter_sums[fighter]['Total_Subs']                 += int(fighter_row['Fighter Submission Attempts'])

                fighter_attr_row += [fight_dur_mins]
                fighter_attr_row += [list(fighter_df_row['Height'])[0], list(fighter_df_row['Reach'])[0], list(fighter_df_row['Stance'])[0]]
                fighter_attr_row += [fighter_sums[fighter]['Total_Sign_Str_Landed'] / fighter_sums[fighter]['Total_Time']]

                if fighter_sums[fighter]['Total_Sign_Str_Attempt'] == 0:
                    fighter_attr_row += [0]
                else:
                    fighter_attr_row += [(fighter_sums[fighter]['Total_Sign_Str_Landed'] / fighter_sums[fighter]['Total_Sign_Str_Attempt']) * 100]

                fighter_attr_row += [fighter_sums[fighter]['Total_Sign_Str_Absorbed'] / fighter_sums[fighter]['Total_Time']]

                if fighter_sums[fighter]['Total_Sign_Str_Opp_Attempt'] == 0:
                    fighter_attr_row += [0]
                else:
                    fighter_attr_row += [(1 - fighter_sums[fighter]['Total_Sign_Str_Absorbed'] / fighter_sums[fighter]['Total_Sign_Str_Opp_Attempt']) * 100]
                
                fighter_attr_row += [fighter_sums[fighter]['Total_TD'] / (fighter_sums[fighter]['Total_Time'] / 15)]
                
                if fighter_sums[fighter]['Total_TD_Attempt'] == 0:
                    fighter_attr_row += [0]
                else: 
                    fighter_attr_row += [(fighter_sums[fighter]['Total_TD'] / fighter_sums[fighter]['Total_TD_Attempt']) * 100]

                if fighter_sums[fighter]['Total_Opp_TD_Attempt'] == 0:
                    fighter_attr_row += [0]
                else:
                    fighter_attr_row += [(1 - fighter_sums[fighter]['Total_Opp_TD'] / fighter_sums[fighter]['Total_Opp_TD_Attempt']) * 100]
                
                fighter_attr_row += [fighter_sums[fighter]['Total_Subs'] / (fighter_sums[fighter]['Total_Time'] / 15)]
            else:
                fighter_attr_row += [fighter_sums[fighter]['Total_Time'] / fighter_sums[fighter]['Matches']]
                fighter_attr_row += [list(fighter_df_row['Height'])[0], list(fighter_df_row['Reach'])[0], list(fighter_df_row['Stance'])[0]]
                fighter_attr_row += [fighter_sums[fighter]['Total_Sign_Str_Landed'] / fighter_sums[fighter]['Total_Time']]
                
                if fighter_sums[fighter]['Total_Sign_Str_Attempt'] == 0:
                    fighter_attr_row += [0]
                else:
                    fighter_attr_row += [fighter_sums[fighter]['Total_Sign_Str_Landed'] / fighter_sums[fighter]['Total_Sign_Str_Attempt']]

                fighter_attr_row += [fighter_sums[fighter]['Total_Sign_Str_Absorbed'] / fighter_sums[fighter]['Total_Time']]

                if fighter_sums[fighter]['Total_Sign_Str_Opp_Attempt'] == 0:
                    fighter_attr_row += [0]
                else:
                    fighter_attr_row += [(1 - fighter_sums[fighter]['Total_Sign_Str_Absorbed'] / fighter_sums[fighter]['Total_Sign_Str_Opp_Attempt']) * 100]
                
                fighter_attr_row += [fighter_sums[fighter]['Total_TD'] / (fighter_sums[fighter]['Total_Time'] / 15)]
                
                if fighter_sums[fighter]['Total_TD_Attempt'] == 0:
                    fighter_attr_row += [0]
                else: 
                    fighter_attr_row += [fighter_sums[fighter]['Total_TD'] / fighter_sums[fighter]['Total_TD_Attempt']]

                if fighter_sums[fighter]['Total_Opp_TD_Attempt'] == 0:
                    fighter_attr_row += [0]
                else:
                    fighter_attr_row += [(1 - fighter_sums[fighter]['Total_Opp_TD'] / fighter_sums[fighter]['Total_Opp_TD_Attempt']) * 100]
                
                fighter_attr_row += [fighter_sums[fighter]['Total_Subs'] / (fighter_sums[fighter]['Total_Time'] / 15)]

                rounds = int(list(total_fight_row['Round'])[0])
                last_round_dur = list(total_fight_row['Time'])[0]
                last_round_dur = (int(last_round_dur.split(':')[0])*60 + int(last_round_dur.split(':')[1])) / 60
                fight_dur_mins = (rounds - 1) * 5 + last_round_dur # Because in all fights the round has 5 mins duration

                fighter_sums[fighter]['Total_Time'] += fight_dur_mins
                
                fighter_sums[fighter]['Total_Sign_Str_Landed']      += int(fighter_row['Fighter Sign.Strikes Done'])
                fighter_sums[fighter]['Total_Sign_Str_Attempt']     += int(fighter_row['Fighter Sign.Strikes Attempted'])
                fighter_sums[fighter]['Total_Sign_Str_Absorbed']    += int(list(opponent_row['Fighter Sign.Strikes Done'])[0])
                fighter_sums[fighter]['Total_Sign_Str_Opp_Attempt'] += int(list(opponent_row['Fighter Sign.Strikes Attempted'])[0])
                fighter_sums[fighter]['Total_TD']                   += int(fighter_row['Fighter Takedowns Done'])
                fighter_sums[fighter]['Total_TD_Attempt']           += int(fighter_row['Fighter Takedowns Attempted'])
                fighter_sums[fighter]['Total_Opp_TD']               += int(opponent_row['Fighter Takedowns Done'])
                fighter_sums[fighter]['Total_Opp_TD_Attempt']       += int(opponent_row['Fighter Takedowns Attempted'])
                fighter_sums[fighter]['Total_Subs']                 += int(fighter_row['Fighter Submission Attempts'])

            result = list(total_fight_row['Result'])[0]

            if result == 'draw':
                fighter_sums[fighter]['Draws'] += 1
            elif result == 'win':
                if list(total_fight_row['Fighter 1 ID'])[0] == fighter_row['Fighter ID']:
                    fighter_sums[fighter]['Wins'] += 1
                elif list(total_fight_row['Fighter 2 ID'])[0] == fighter_row['Fighter ID']:
                    fighter_sums[fighter]['Loses'] += 1
                else:
                   print("I got wrong fighter ID ",row['Fighter ID'])
            elif result == 'lose':
                if list(total_fight_row['Fighter 1 ID'])[0] == fighter_row['Fighter ID']:
                    fighter_sums[fighter]['Loses'] += 1
                elif list(total_fight_row['Fighter 2 ID'])[0] == fighter_row['Fighter ID']:
                    fighter_sums[fighter]['Wins'] += 1
                else:
                    print("I got wrong fighter ID ",row['Fighter ID'])
            elif result == 'no contest':
                # do nothing
                fighter_sums[fighter]['Wins'] += 0
            else:
                print("I got wrong result ", result)

            fighter_sums[fighter]['Matches'] += 1

            assert fighter_attr_row[0] not in fighter_fight_attrs[fighter].keys()

            fighter_fight_attrs[fighter][fighter_attr_row[0]] = fighter_attr_row

    create_final_df(init_event_df, fighter_fight_attrs, './Final_Fights_Dataset.csv')

if __name__ == "__main__":
    main()