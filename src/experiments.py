import sys

sys.path.append('./Utils')
sys.path.append('./Processes')

import numpy  as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.svm          import SVC
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import RandomForestClassifier, AdaBoostClassifier

from utils import read_fights_data, df_get_na

from preprocessing_utils import double_dataset, fighter_stats_diff_dataset

from train_utils import RepeatedKFoldCrossValidation, ClassifierGridSearch

from preprocessing import basic_preprocessing, eda_preprocessing
from preprocessing import dim_reduction_preprocessing, before_train_preprocessing

from eda import dim_reduction_eda

from train    import trainClassifierAndPlotResults, evaluateClassifiersTraining
from train    import evaluateBestClassifiers
from validate import evaluateClassifierProduction



# Just set their hyperparameters
def prepare_best_classifiers_for_training():
    clf_refs = [LogisticRegression, KNeighborsClassifier, SVC, DecisionTreeClassifier,
                RandomForestClassifier, AdaBoostClassifier]
    
    logistic_regression_params = {
        'C': 1.0,
        'max_iter': 10000
    }

    knn_params = {
        'n_neighbors': 75
    }

    svm_params = {
        'C': 1.0,
    }

    decision_tree_params = {
        'max_depth': 15
    }

    random_forest_params = {
        'n_estimators': 150
    }

    ada_boost_params = {
        'n_estimators': 100
    }

    clf_params_list = [logistic_regression_params, knn_params, svm_params,
                        decision_tree_params, random_forest_params, ada_boost_params]

    return clf_refs, clf_params_list



# Just set their hyperparameters
def prepare_classifiers_for_training():
    clf_refs = [LogisticRegression, KNeighborsClassifier, SVC, DecisionTreeClassifier,
                RandomForestClassifier, AdaBoostClassifier]
    
    logistic_regression_params = {
        'C': [0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0],
        'max_iter': [10000]
    }

    knn_params = {
        'n_neighbors': [3, 5, 10, 20, 50, 75, 100, 150, 200, 250]
    }

    svm_params = {
        'C': [0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0],
    }

    decision_tree_params = {
        'max_depth': [5, 10, 15, 20, 25, 50]
    }

    random_forest_params = {
        'n_estimators': [50, 75, 100, 150, 200, 300]
    }

    ada_boost_params = {
        'n_estimators': [50, 75, 100, 150, 200, 300]
    }

    clf_params_list = [logistic_regression_params, knn_params, svm_params,
                        decision_tree_params, random_forest_params, ada_boost_params]

    return clf_refs, clf_params_list



def main():
    X, Y = read_fights_data('./data/Fights.csv')
    X, Y = basic_preprocessing(X, Y)
    y    = Y['Result']

    # Keeping the last 500 fights for validation
    X_validation = X.iloc[len(X) - 500:len(X), :]
    y_validation = y.iloc[len(y) - 500:len(y)]
    print(y_validation.value_counts())
    X = X.iloc[0:len(X)-500, :]
    y = y.iloc[0:len(y)-500]

    print(X['Fighter_1_Stance'].unique())
    print('Columns of label dataframe:', Y.columns)

    print(X.head(5))

    # Getting the other forms of the dataset that we will do
    # Dimensionality Reduction
    X_og,           y_og           = X, y
    X_doubled,      y_doubled      = double_dataset(X_og, y_og)
    X_diff,         y_diff         = fighter_stats_diff_dataset(X),         pd.Series(y)
    X_doubled_diff, y_doubled_diff = fighter_stats_diff_dataset(X_doubled), pd.Series(y_doubled)

    dr_X_og           = dim_reduction_preprocessing(X_og)
    dr_X_doubled      = dim_reduction_preprocessing(X_doubled)
    dr_X_diff         = dim_reduction_preprocessing(X_diff)
    dr_X_doubled_diff = dim_reduction_preprocessing(X_doubled_diff)

    dim_reduction_eda(dr_X_og, y_og, dr_X_doubled, y_doubled, dr_X_diff, y_diff,
                    dr_X_doubled_diff, y_doubled_diff)

    # Getting ready to train and test
    train_X, train_y = pd.DataFrame(X), pd.Series(y)
    
    # Evaluating each classifier with each hyperparameters

    clf_refs, clf_params_list = prepare_classifiers_for_training()
    evaluateClassifiersTraining(clf_refs, clf_params_list,
                            train_X, train_y, 6, 2,
                            after_split_preprocessing_fn=before_train_preprocessing)
    
    # Evaluate the best of the best in different versions of the dataset

    best_clf_refs, best_clf_params = prepare_best_classifiers_for_training()
    evaluateBestClassifiers(best_clf_refs, best_clf_params, train_X, train_y,
                            X_validation, y_validation,
                            5, 2, before_train_preprocessing)

    best_clf_refs, best_clf_params = prepare_best_classifiers_for_training()
    evaluateBestClassifiers(best_clf_refs, best_clf_params, train_X, train_y,
                            X_validation, y_validation,
                            5, 2, before_train_preprocessing, {'to_double': True})

    best_clf_refs, best_clf_params = prepare_best_classifiers_for_training()
    evaluateBestClassifiers(best_clf_refs, best_clf_params, train_X, train_y,
                            X_validation, y_validation,
                            5, 2, before_train_preprocessing, {'to_double': True, 'to_diff': True})
    #print(X_double)
    #print(X_diff)
    #print(X_double_diff)

    return 0

if __name__ == '__main__':
    main()
