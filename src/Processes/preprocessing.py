# Preprocessing functions that contain the preprocessing
# that should be done before some processes, such as
# EDA, Dimensionality Reduction, Train, etc.
# Note that some preprocessing functions should be called
# sequentially.

import sys

sys.path.append('../Utils')

import numpy  as np
import pandas as pd
from   sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import df_get_na
from preprocessing_utils import series_convert_feet_inches_to_cm, keep_valid_result_rows
from preprocessing_utils import series_fillna_most_frequent, series_fillna_mean
from preprocessing_utils import series_categorical_to_int, df_drop_columns
from preprocessing_utils import double_dataset, fighter_stats_diff_dataset

# In that function I will just do pretty basic preprocessing needed
# to do EDA decently.
def basic_preprocessing(X, Y=None):
    print('NaN Values')
    print(df_get_na(X))

    # Keeping only fights with result equal to win, lose or draw(if we provide Y)
    if Y is not None:
        X, Y = keep_valid_result_rows(X, Y)

    # Converting height and reach to cm.
    X['Fighter_1_Height'] = series_convert_feet_inches_to_cm(X['Fighter_1_Height'])
    X['Fighter_1_Reach']  = series_convert_feet_inches_to_cm(X['Fighter_1_Reach'])

    # Removing NaN values
    X['Fighter_1_Height'] = series_fillna_mean(X['Fighter_1_Height'])
    X['Fighter_1_Reach'] = series_fillna_mean(X['Fighter_1_Reach'])

    X['Fighter_2_Height'] = series_fillna_mean(series_convert_feet_inches_to_cm(X['Fighter_2_Height']))
    X['Fighter_2_Reach'] = series_fillna_mean(series_convert_feet_inches_to_cm(X['Fighter_2_Reach']))

    X['Fighter_1_Age'] = series_fillna_mean(X['Fighter_1_Age'])
    X['Fighter_2_Age'] = series_fillna_mean(X['Fighter_2_Age'])

    X['Fighter_1_Stance'] = series_fillna_most_frequent(X['Fighter_1_Stance'])
    X['Fighter_2_Stance'] = series_fillna_most_frequent(X['Fighter_2_Stance'])

    print()
    print('New NaN Values')
    print(df_get_na(X))

    # Lowercase every str or categorical column that is not case sensitive
    str_columns = ['Gender', 'Weight_Class', 'Fight_Time_Format', 'Fighter_1_Name', 
                    'Fighter_1_Nickname', 'Fighter_1_Stance', 'Fighter_2_Name', 
                    'Fighter_2_Nickname', 'Fighter_2_Stance']
    
    for col in str_columns:
        X[col] = X[col].apply(lambda x: x.lower())

    if Y is not None:
        return X, Y
    else: 
        return X



def eda_preprocessing(X, Y):
    new_X = pd.DataFrame(X)
    new_Y = pd.DataFrame(Y)

    # I only need to split the fight date into 3 features
    new_X['Fight_Date'] = pd.to_datetime(new_X['Fight_Date'])

    fight_year_series  = new_X['Fight_Date'].dt.year
    fight_month_series = new_X['Fight_Date'].dt.month
    fight_day_series   = new_X['Fight_Date'].dt.day

    new_X.insert(1, 'Fight_Year',  fight_year_series)
    new_X.insert(1, 'Fight_Month', fight_month_series)
    new_X.insert(1, 'Fight_Day',   fight_day_series)
    
    new_X = df_drop_columns(new_X, ['Fight_Date'])

    return new_X, new_Y



def _cat_attrs_to_int(X):
    # Converting categorical features to numerical
    # Title_Fight, Weight_Class and Gender are the categorical columns
    # (the names and the nicknames probably might be removed)
    """
    print(X['Title_Fight'].unique())
    print(X['Weight_Class'].unique())
    print(X['Gender'].unique())
    print(X['Fight_Time_Format'].unique())
    print(X['Fighter_1_Stance'].unique())
    print(X['Fighter_2_Stance'].unique())
    """
    
    X['Title_Fight']  = series_categorical_to_int(X['Title_Fight'], {False: 0, True: 1})
    X['Gender']       = series_categorical_to_int(X['Gender'], {'female': 0, 'male': 1})

    # Need to have weight classes in an ordinal manner
    weight_class_map = {'catch weight': 9, 'heavyweight': 8, 'light heavyweight': 7, 
                        'middleweight': 6, 'welterweight': 5, 'lightweight': 4,
                        'featherweight': 3, 'bantamweight': 2, 'flyweight': 1, 
                        'strawweight': 0}

    X['Weight_Class'] = series_categorical_to_int(X['Weight_Class'], weight_class_map)
    
    X['Fight_Time_Format'] = series_categorical_to_int(X['Fight_Time_Format'], 
                                            {'3rnd(5-5-5)': 0, '5rnd(5-5-5-5-5)': 1})

    # Need to convert the stance of each fighter
    stance_map = {'orthodox': 0, 'southpaw': 1, 'open stance': 2,
                'switch': 3, 'sideways': 4}

    X['Fighter_1_Stance'] = series_categorical_to_int(X['Fighter_1_Stance'], stance_map)
    X['Fighter_2_Stance'] = series_categorical_to_int(X['Fighter_2_Stance'], stance_map)

    return



def dim_reduction_preprocessing(X):
    # Dropping unnecessary columns
    new_X = df_drop_columns(X, ['Fight_ID', 'Fighter_1_ID', 'Fighter_1_Name', 
                            'Fighter_1_Nickname', 'Fighter_2_ID',
                            'Fighter_2_Name', 'Fighter_2_Nickname'])

    # Splitting dates
    new_X['Fight_Date']= pd.to_datetime(new_X['Fight_Date'])

    fight_year_series  = new_X['Fight_Date'].dt.year
    fight_month_series = new_X['Fight_Date'].dt.month
    fight_day_series   = new_X['Fight_Date'].dt.day

    new_X.insert(0, 'Fight_Year',  fight_year_series)
    new_X.insert(0, 'Fight_Month', fight_month_series)
    new_X.insert(0, 'Fight_Day',   fight_day_series)
    
    new_X = df_drop_columns(new_X, ['Fight_Date'])

    # Scaling each numerical column, except the ones that denote percentage
    scaler = MinMaxScaler()

    categorical_attrs = ['Gender', 'Title_Fight', 'Weight_Class', 'Fight_Time_Format', 
                        'Fighter_1_Stance', 'Fighter_2_Stance']

    percentage_attrs  = ['Fighter_1_Str_Acc', 'Fighter_1_Defense', 'Fighter_1_Takedown_Acc',
                        'Fighter_1_Takedown_Def', 'Fighter_2_Str_Acc', 'Fighter_2_Defense', 
                        'Fighter_2_Takedown_Acc', 'Fighter_2_Takedown_Def']

    for attr in new_X.columns:
        if (attr not in categorical_attrs) and (attr not in percentage_attrs):
            scale_vals_len = len(new_X[attr])

            scale_vals = np.reshape(new_X[attr].values, (-1, 1))
            scale_vals = scaler.fit_transform(scale_vals)

            new_X[attr] = np.reshape(scale_vals, (scale_vals_len, ))

    # Converting categoricals to numericals
    _cat_attrs_to_int(new_X)

    print(new_X)
    return new_X




def _test_train_common_preprocessing(X, y):
    # Dropping unnecessary columns
    new_X = df_drop_columns(X, ['Fight_ID', 'Fighter_1_ID', 'Fighter_1_Name', 
                                'Fighter_1_Nickname', 'Fighter_2_ID',
                                'Fighter_2_Name', 'Fighter_2_Nickname'])

    new_y = pd.Series(y) if y is not None else None

    # Splitting dates for train and test set
    new_X['Fight_Date']= pd.to_datetime(new_X['Fight_Date'])

    #fight_year_series  = new_X['Fight_Date'].dt.year
    #fight_month_series = new_X['Fight_Date'].dt.month
    #fight_day_series   = new_X['Fight_Date'].dt.day

    #new_X.insert(0, 'Fight_Year',  fight_year_series)
    #new_X.insert(0, 'Fight_Month', fight_month_series)
    #new_X.insert(0, 'Fight_Day',   fight_day_series)
    
    new_X = df_drop_columns(new_X, ['Fight_Date'])

    # Converting categoricals to numericals
    _cat_attrs_to_int(new_X)

    return new_X, new_y



# This should be done right before train, so this is done after
# Train-Test split. The test instances might not have labels in
# case of real time testing.
def before_train_preprocessing(X_train, y_train, X_test, y_test=None, to_double=False, to_diff=False):
    new_train_X = pd.DataFrame(X_train)
    new_train_y = pd.Series(y_train)

    new_test_X  = pd.DataFrame(X_test)
    new_test_y  = pd.Series(y_test) if y_test is not None else None

    if to_double == True:
        new_train_X, new_train_y = double_dataset(X_train, y_train)
    
    if to_diff == True:
        new_train_X = fighter_stats_diff_dataset(new_train_X)
        new_test_X  = fighter_stats_diff_dataset(new_test_X)

    new_train_X, new_train_y = _test_train_common_preprocessing(new_train_X, new_train_y)
    new_test_X,  new_test_y  = _test_train_common_preprocessing(new_test_X,  new_test_y)

    # Scaling each numerical column, except percentage features
    scaler = MinMaxScaler()

    categorical_attrs = ['Gender', 'Title_Fight', 'Weight_Class', 'Fight_Time_Format', 
                        'Fighter_1_Stance', 'Fighter_2_Stance']

    percentage_attrs  = ['Fighter_1_Str_Acc', 'Fighter_1_Defense', 'Fighter_1_Takedown_Acc',
                        'Fighter_1_Takedown_Def', 'Fighter_2_Str_Acc', 'Fighter_2_Defense', 
                        'Fighter_2_Takedown_Acc', 'Fighter_2_Takedown_Def']

    assert(list(new_train_X.columns) == list(new_test_X.columns))

    for attr in new_train_X.columns:
        if (attr not in categorical_attrs) and (attr not in percentage_attrs):
            train_scale_vals_len = len(new_train_X[attr])
            test_scale_vals_len  = len(new_test_X[attr])

            train_scale_vals = np.reshape(new_train_X[attr].values, (-1, 1))
            test_scale_vals  = np.reshape(new_test_X[attr].values,  (-1, 1))

            scaler.fit(train_scale_vals)

            train_scale_vals = scaler.transform(train_scale_vals)
            test_scale_vals  = scaler.transform(test_scale_vals)

            new_train_X[attr] = np.reshape(train_scale_vals, (train_scale_vals_len, ))
            new_test_X[attr]  = np.reshape(test_scale_vals,  (test_scale_vals_len, ))

    if new_test_y is not None:
        return new_train_X, new_train_y, new_test_X, new_test_y
    else:
        return new_train_X, new_train_y, new_test_X
