import os
import sys

sys.path.append('./Utils')
sys.path.append('./Processes')

from datetime import datetime
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import AdaBoostClassifier, RandomForestClassifier

from utils         import read_fights_data, read_fighters_data, read_matchup_data
from preprocessing import basic_preprocessing, before_train_preprocessing



mma_weight_classes = ['atomweight', 'strawweight', 'flyweight', 'bantamweight', 'featherweight', 'lightweight',
					'super lightweight', 'welterweight', 'super welterweight', 'middleweight', 'super middleweight',
					'light heavyweight', 'cruiserweight', 'heavyweight']



def extract_args(argv):
    if len(argv) != 6:
        print("Invalid number of arguments")
        print("Usage: python3 demo.py -t <train_dataset_path> -f <fighters_df_path> -p <prediction_dataset_path>")

        return None, None, None
    
    train_dataset_path    = None
    fighters_dataset_path = None
    matchups_dataset_path = None

    for i in range(0, 6, 2):
        curr_arg = argv[i]
        arg_val  = argv[i + 1]

        if not os.path.isfile(arg_val):
            print("File " + arg_val + " does not exist or it is not a file.")
            print("Usage: python3 demo.py -t <train_dataset_path> -f <fighters_df_path> -p <prediction_dataset_path>")

            return None, None, None

        if curr_arg == '-t':
            if train_dataset_path is not None:
                print("You already specified training dataset's path")
                print("Usage: python3 demo.py -t <train_dataset_path> -f <fighters_df_path> -p <prediction_dataset_path>")

                return None, None, None

            train_dataset_path = arg_val    
        elif curr_arg == '-f':
            if fighters_dataset_path is not None:
                print("You already specified fighters dataset's path")
                print("Usage: python3 demo.py -t <train_dataset_path> -f <fighters_df_path> -p <prediction_dataset_path>")

                return None, None, None

            fighters_dataset_path = arg_val  
        elif curr_arg == '-p':
            if matchups_dataset_path is not None:
                print("You already specified matchups dataset's path")
                print("Usage: python3 demo.py -t <train_dataset_path> -f <fighters_dataset_path> -p <matchups_dataset_path>")

                return None, None, None

            matchups_dataset_path = arg_val 
        else:
            print("Invalid argument " + curr_arg)
            print("Usage: python3 demo.py -t <train_dataset_path> -f <fighters_df_path> -p <prediction_dataset_path>")

            return None, None, None

    return train_dataset_path, fighters_dataset_path, matchups_dataset_path



# Validation of the matchups that the user gave
def validate_matchup_df(matchups_df, fighters_df):
    if len(matchups_df.columns) != 5:
        print('Match-up dataset should have 5 columns')
        return False

    matchup_weight_classes = list(matchups_df['Weight Class'])

    # Check that user gave valid weight classes
    for weight_class in matchup_weight_classes:
        if weight_class not in mma_weight_classes:
            print('Some matchup has an invalid weight class')
            return False
    
    # Check that in title fight field he gave only True/False
    for rounds in list(matchups_df['Title Fight']):
        if rounds not in [True, False]:
            print('You should specify if the fight is title fight with True or False')
            return False
    
    # Check that he wrote 3 or 5 in rounds field
    for rounds in list(matchups_df['Rounds']):
        if rounds not in [3, 5]:
            print('In the rounds field you can only give 3 or 5')
            return False         

    # Check that fights are of the same gender
    fighter_1_genders = list(matchups_df.merge(pd.merge(matchups_df, fighters_df, how='inner', left_on=['Fighter 1 ID'],
                                right_on=['Fighter ID']))['Gender'])

    fighter_2_genders = list(matchups_df.merge(pd.merge(matchups_df, fighters_df, how='inner', left_on=['Fighter 2 ID'],
                                right_on=['Fighter ID']))['Gender'])

    if fighter_1_genders != fighter_2_genders:
        print('You have a matchup between fighters of different gender')
        return False

    return True



# Creation of training vectors as they should be in order to apply preprocessing
# and training
def create_matchup_features(matchups_df, fighters_df):
    today_date  = datetime.today().strftime('%Y-%m-%d')
    today_year  = int(today_date.split('-')[0])

    new_columns = ['Fight_ID', 'Fight_Date', 'Gender', 'Weight_Class', 'Title_Fight', 'Fight_Time_Format', 'Fighter_1_ID', 
        'Fighter_1_Name', 'Fighter_1_Nickname', 'Fighter_1_Age', 'Fighter_1_Wins', 'Fighter_1_Loses', 'Fighter_1_Draws', 'Fighter_1_Avg_Time(MINS)', 
        'Fighter_1_Height', 'Fighter_1_Reach', 'Fighter_1_Stance', 'Fighter_1_Sign_SLpMin', 'Fighter_1_Str_Acc', 'Fighter_1_Sign_SApMin', 
        'Fighter_1_Defense', 'Fighter_1_Takedown_Avgp15M', 'Fighter_1_Takedown_Acc', 'Fighter_1_Takedown_Def', 'Fighter_1_Sub_Avgp15M', 
        'Fighter_2_ID', 'Fighter_2_Name', 'Fighter_2_Nickname', 'Fighter_2_Age', 'Fighter_2_Wins', 'Fighter_2_Loses', 'Fighter_2_Draws', 
        'Fighter_2_Avg_Time(MINS)', 'Fighter_2_Height', 'Fighter_2_Reach', 'Fighter_2_Stance', 'Fighter_2_Sign_SLpMin', 'Fighter_2_Str_Acc', 
        'Fighter_2_Sign_SApMin', 'Fighter_2_Defense', 'Fighter_2_Takedown_Avgp15M', 'Fighter_2_Takedown_Acc', 'Fighter_2_Takedown_Def', 
        'Fighter_2_Sub_Avgp15M']

    # I'll just add adhoc IDs, dates and nicknames as they will be dropped afterwards
    fight_ids     = pd.Series(list(range(len(matchups_df))))
    fight_dates   = pd.Series([datetime.today().strftime('%Y-%m-%d')] * len(matchups_df))
    fighter_nicks = pd.Series(['No_Nickname'] * len(matchups_df))

    fighter_1_df = matchups_df.merge(pd.merge(matchups_df, fighters_df, how='inner', left_on=['Fighter 1 ID'],
                                right_on=['Fighter ID']))[fighters_df.columns]

    fighter_2_df = matchups_df.merge(pd.merge(matchups_df, fighters_df, how='inner', left_on=['Fighter 2 ID'],
                                right_on=['Fighter ID']))[fighters_df.columns]

    # Getting fighters age at time of fight
    fighter_1_age = fighter_1_df['DOB'].apply(lambda x: today_year - int(x.split(' ')[2]))
    fighter_2_age = fighter_2_df['DOB'].apply(lambda x: today_year - int(x.split(' ')[2]))

    # Converting rounds to fight time format
    fight_time_format = matchups_df['Rounds'].apply(lambda x: str(x) + 'Rnd(' + '-'.join(['5'] * int(x)) + ')')

    # Creating the validation dataset
    
    # Inserting the common attributes columns
    validation_df = pd.concat([fight_ids, fight_dates, fighter_1_df['Gender'], matchups_df['Weight Class'],
                                matchups_df['Title Fight'], fight_time_format], axis=1)
    
    # Inserting First fighter's attributes
    validation_df = pd.concat([validation_df, fighter_1_df['Fighter ID'], fighter_1_df['Fighter Name'], fighter_nicks, 
                                fighter_1_age, fighter_1_df['Wins'], fighter_1_df['Loses'],
                                fighter_1_df['Draws'], fighter_1_df['Avg.Time(in Mins)']], axis=1)

    validation_df = pd.concat([validation_df, fighter_1_df['Height'], fighter_1_df['Reach'],
                                fighter_1_df['Stance'], fighter_1_df['SLpM'],
                                fighter_1_df['Str.Acc.'], fighter_1_df['SApM'],
                                fighter_1_df['Str. Def.'], fighter_1_df['TD Avg.'],
                                fighter_1_df['TD Acc.'], fighter_1_df['TD Def.'],
                                fighter_1_df['Sub. Avg.']], axis=1)

    # Inserting Second fighter's attributes
    validation_df = pd.concat([validation_df, fighter_2_df['Fighter ID'], fighter_2_df['Fighter Name'], fighter_nicks, 
                                fighter_2_age, fighter_2_df['Wins'], fighter_2_df['Loses'],
                                fighter_2_df['Draws'], fighter_2_df['Avg.Time(in Mins)']], axis=1)

    validation_df = pd.concat([validation_df, fighter_2_df['Height'], fighter_2_df['Reach'],
                                fighter_2_df['Stance'], fighter_2_df['SLpM'],
                                fighter_2_df['Str.Acc.'], fighter_2_df['SApM'],
                                fighter_2_df['Str. Def.'], fighter_2_df['TD Avg.'],
                                fighter_2_df['TD Acc.'], fighter_2_df['TD Def.'],
                                fighter_2_df['Sub. Avg.']], axis=1)

    validation_df.columns = new_columns

    return validation_df



def train_model(fights_data, fights_labels):
    # The best model found in our experiments is a LogisticRegression model
    model = RandomForestClassifier(n_estimators=150)
    model.fit(fights_data, fights_labels)

    return model



def make_and_output_preds(model, matchups_df, names_df):
    classes = model.classes_

    preds = model.predict_proba(matchups_df).tolist()

    # Setting up the prediction probabilities as dictionary
    preds_dicts = []

    for pred in preds:
        curr_dict = {}

        for curr_class, prob in zip(classes, pred):
            curr_dict[curr_class] = round(prob * 100, 2)

        preds_dicts.append(curr_dict)

    preds = preds_dicts

    # Printing the predictions
    print('The predictions are the following:')

    for i, pred, iterrow in zip(range(1, len(preds) + 1), preds, names_df.iterrows()):
        _, names = iterrow

        print(i, '.', names['Fighter_1_Name'], 'vs', names['Fighter_2_Name'])
        
        win_prefix  = names['Fighter_1_Name']
        lose_prefix = names['Fighter_2_Name']
        
        for curr_class, prefix in zip(['win', 'lose', 'draw'], [win_prefix, lose_prefix, 'draw']):
            print('\t' + prefix + ':', pred[curr_class])
        
        print()

    return



def main(train_dataset_path, fighters_dataset_path, matchups_path):
    # Reading the necessary datasets
    fights_df, fights_labels = read_fights_data(train_dataset_path) 
    fighters_df              = read_fighters_data(fighters_dataset_path)
    matchups_df              = read_matchup_data(matchups_path)

    # Configuring matchups and get the names to a seperate dataframe
 
    if validate_matchup_df(matchups_df, fighters_df) == True:
        prediction_df = create_matchup_features(matchups_df, fighters_df)
    else:
        return 1

    names_df      = pd.concat([prediction_df['Fighter_1_Name'], prediction_df['Fighter_1_Nickname']], axis=1)
    names_df      = pd.concat([names_df, prediction_df['Fighter_2_Name'], prediction_df['Fighter_2_Nickname']], axis=1)

    # Preprocessing the fights dataset and the validation one.
    fights_df, fights_labels = basic_preprocessing(fights_df, fights_labels)
    prediction_df            = basic_preprocessing(prediction_df)

    fights_labels = fights_labels['Result']

    fights_df, fights_labels, prediction_df = before_train_preprocessing(fights_df, fights_labels, 
                                                        prediction_df, to_double=True, to_diff=True)

    # Training our model
    model = train_model(fights_df, fights_labels)

    # Making the predictions and print them
    make_and_output_preds(model, prediction_df, names_df)

    return 0

"""
Plan for execution(it will run on the best classifier)
python3 demo.py -t <train_dataset_path> -f <fighters_df_path> -p <prediction_dataset_path>
where: 
    1. Training set will be always be the ./data/Fights.csv and the dataset with the fighters will
    be the ./data/Fighters.csv.

    2. -p argument has the path of the dataset that we will make predictions. This dataset should be a csv
    file with the following columns(the columns should be pipe-separated):
        -> Weight Class
        -> Title Fight
        -> Rounds(should be 3 or 5)
        -> Fighter 1 ID(can be taken from Fighters.csv)
        -> Fighter 2 ID
    
    and the attributes should have the following constraints:
        -> The fighter's should be of the same gender(this will be taken from their last fight)
        -> The weightclass should be valid
"""
if __name__ == '__main__':
    train_dataset_path, fighters_dataset_path, matchups_dataset_path = extract_args(sys.argv[1:])

    if train_dataset_path is not None:
        main(train_dataset_path, fighters_dataset_path, matchups_dataset_path)
