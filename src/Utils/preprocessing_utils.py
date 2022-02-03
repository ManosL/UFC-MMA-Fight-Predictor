import pandas as pd
import numpy  as np
import re

inches_feet_height_re = re.compile(r"(?:([0-9]+)')?\s*([0-9]*\.?[0-9]+)\"")
def feet_inches_to_cm(val):
    if val is np.nan:
        return np.nan

    m = inches_feet_height_re.match(val)

    if m == None:
        return np.nan
    else:
        feet   = int(m.group(1)) if m.group(1) != None else 0
        inches = float(m.group(2))

        return int((feet * 12 + inches) * 2.54)



# Converts the height from feet & inches to cm
def series_convert_feet_inches_to_cm(height_series):
    return height_series.apply(lambda x: feet_inches_to_cm(x))



# Series fillna with most frequent
def series_fillna_most_frequent(series):
    return series.fillna(series.value_counts().index[0])



# Series fillna with mean
def series_fillna_mean(series, inplace_m=False):
    return series.fillna(series.mean())



def series_categorical_to_int(series, map_dict=None):
    if map_dict == None:
        map_dict    = {}
        unique_vals = series.unique()

        for i, key in zip(range(len(unique_vals)), unique_vals):
            map_dict[key] = i

    return series.map(map_dict)



# Keep rows that have as a result Win/Lose/Draw
def keep_valid_result_rows(X, Y):
    new_X = pd.DataFrame(X[Y['Result'].isin(['win', 'lose', 'draw'])])
    new_Y = pd.DataFrame(Y[Y['Result'].isin(['win', 'lose', 'draw'])])
    
    new_X.reset_index(drop=True, inplace=True)
    new_Y.reset_index(drop=True, inplace=True)
    
    print(new_X)
    print(new_Y)
    return new_X, new_Y



def _reverse_result(res):
    res_low = res.lower()

    if res_low == 'win':
        return 'lose'
    elif res_low == 'lose':
        return 'win'
    elif res_low == 'draw':
        return 'draw'
    elif res_low == 'no contest':
        return 'no contest'
    
    assert(0 == 1)
    return None



def df_drop_columns(X, cols):
    return X.drop(cols, axis=1)



# This function just doubles the original dataset size by adding the symmetric
# rows.
# y is a vector that has the classes "win", "lose" or "draw"
# This function assumes that X has the full matchup of the fights
# and it is not called the fighter_stats_diff function before.
def double_dataset(X, y):
    doubled_X_part_1 = pd.DataFrame(X)
    double_y_part_1  = pd.Series(y)

    doubled_X_part_2 = pd.concat([doubled_X_part_1.iloc[:, 0:6], doubled_X_part_1.iloc[:, 25:44], 
                                doubled_X_part_1.iloc[:, 6:25]], axis=1, ignore_index=True)
    
    doubled_X_part_2.columns = doubled_X_part_1.columns
    double_y_part_2 = double_y_part_1.apply(lambda x: _reverse_result(x))

    doubled_X = pd.concat([doubled_X_part_1, doubled_X_part_2], axis=0, ignore_index=True)
    doubled_y = pd.concat([double_y_part_1, double_y_part_2], axis=0, ignore_index=True)

    return doubled_X, doubled_y



# This function keeps the difference between fighter's each attribute
# on the matchup of each fight. The new attrs will have the form
# <Fighter 1 attr - Fighter 2 attr>
def fighter_stats_diff_dataset(X):
    """
    Initial columns

    ['Fight_ID', 'Fight_Date', 'Gender', 'Weight_Class', 'Title_Fight',
        'Fight_Time_Format', 'Fighter_1_ID', 'Fighter_1_Name', 'Fighter_1_Nickname',
        'Fighter_1_Age', 'Fighter_1_Wins', 'Fighter_1_Loses', 'Fighter_1_Draws',
        'Fighter_1_Avg_Time(MINS)', 'Fighter_1_Height', 'Fighter_1_Reach',
        'Fighter_1_Stance', 'Fighter_1_Sign_SLpMin', 'Fighter_1_Str_Acc',
        'Fighter_1_Sign_SApMin', 'Fighter_1_Defense',
        'Fighter_1_Takedown_Avgp15M', 'Fighter_1_Takedown_Acc',
        'Fighter_1_Takedown_Def', 'Fighter_1_Sub_Avgp15M', 'Fighter_2_ID', 'Fighter_2_Name',
        'Fighter_2_Nickname', 'Fighter_2_Age', 'Fighter_2_Wins',
        'Fighter_2_Loses', 'Fighter_2_Draws', 'Fighter_2_Avg_Time(MINS)',
        'Fighter_2_Height', 'Fighter_2_Reach', 'Fighter_2_Stance',
        'Fighter_2_Sign_SLpMin', 'Fighter_2_Str_Acc', 'Fighter_2_Sign_SApMin',
        'Fighter_2_Defense', 'Fighter_2_Takedown_Avgp15M',
        'Fighter_2_Takedown_Acc', 'Fighter_2_Takedown_Def',
        'Fighter_2_Sub_Avgp15M']
    """

    new_columns = ['Fight_ID', 'Fight_Date', 'Gender', 'Weight_Class', 'Title_Fight',
        'Fight_Time_Format', 'Fighter_1_ID', 'Fighter_1_Name', 'Fighter_1_Nickname', 
        'Fighter_1_Stance', 'Fighter_2_ID', 'Fighter_2_Name', 'Fighter_2_Nickname', 
        'Fighter_2_Stance', 'Age_Difference', 'Wins_Difference', 'Loses_Difference', 
        'Draws_Difference', 'Avg_Time(MINS)_Difference', 'Height_Difference', 
        'Reach_Difference', 'Sign_SLpMin_Difference', 'Str_Acc_Difference',
        'Sign_SApMin_Difference', 'Defense_Difference', 'Takedown_Avgp15M_Difference', 
        'Takedown_Acc_Difference', 'Takedown_Def_Difference', 'Sub_Avgp15M_Difference']
    
    common_attrs_X  = pd.concat([X.iloc[:,0:9], X['Fighter_1_Stance'],
                                X['Fighter_2_Name'], X['Fighter_2_ID'], X['Fighter_2_Nickname'],
                                X['Fighter_2_Stance']], axis=1, ignore_index=True)

    # The "middle" one is the Stance which we wont need in the substraction
    fighter_1_attrs = pd.concat([X.iloc[:,9:16], X.iloc[:,17:25]],  axis=1, ignore_index=True)
    fighter_2_attrs = pd.concat([X.iloc[:,28:35], X.iloc[:,36:44]], axis=1, ignore_index=True)

    assert(fighter_1_attrs.shape == fighter_2_attrs.shape)
    
    diff_X = fighter_1_attrs.subtract(fighter_2_attrs)
    diff_X = pd.concat([common_attrs_X, diff_X], axis=1, ignore_index=True)

    diff_X.columns = new_columns

    return diff_X
