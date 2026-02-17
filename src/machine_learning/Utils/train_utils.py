import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score

import itertools



# This function will take as arguments the classifier's reference
# of the classifier that we want to evaluate, its parameters in a
# dict of form {param_name}->{list of possible values}, and the 
# RepeatedKFoldCrossValidation args
def ClassifierGridSearch(clf_ref, clf_params, X, y, k=10, repeats=2, 
                        after_split_preprocessing_fn=None, kwargs={}):

    result_dict = {
        'clf_parameters':      [],
        'train_mean_accuracy': [],
        'train_accuracy_std':  [],
        'test_mean_accuracy':  [],
        'test_accuracy_std':   [],
        'train_mean_f1_score': [],
        'train_f1_score_std':  [],
        'test_mean_f1_score':  [],
        'test_f1_score_std':   []
    }

    param_names = clf_params.keys()
    params_vals = clf_params.values()

    # For each combination of possible parameters
    for curr_clf_params in itertools.product(*params_vals):
        curr_kwargs = {}

        for param_name, param_val in zip(param_names, curr_clf_params):
            curr_kwargs[param_name] = param_val
        
        clf = clf_ref(**curr_kwargs)

        results = RepeatedKFoldCrossValidation(k, repeats, clf, X, y,
                                        after_split_preprocessing_fn, kwargs)

        train_mean_acc, train_acc_std, test_mean_acc, test_acc_std,\
        train_mean_f1, train_f1_std, test_mean_f1, test_f1_std      = results

        result_dict['clf_parameters'].append(curr_kwargs)
        result_dict['train_mean_accuracy'].append(train_mean_acc)
        result_dict['train_accuracy_std'].append(train_acc_std)
        result_dict['test_mean_accuracy'].append(test_mean_acc)
        result_dict['test_accuracy_std'].append(test_acc_std)
        result_dict['train_mean_f1_score'].append(train_mean_f1)
        result_dict['train_f1_score_std'].append(train_f1_std)
        result_dict['test_mean_f1_score'].append(test_mean_f1)
        result_dict['test_f1_score_std'].append(test_f1_std)

        print('Done', clf_ref.__name__, curr_kwargs)
    
    return result_dict



# clf should be already initialized
def RepeatedKFoldCrossValidation(k, repeats, clf, X, y, after_split_preprocessing_fn=None, kwargs={}):
    train_accuracies = []
    train_f1_scores  = []
    test_accuracies  = []
    test_f1_scores   = []

    repeated_k_fold = RepeatedKFold(n_splits=k, n_repeats=repeats)

    for train_index, test_index in repeated_k_fold.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # In case we need to do any preprocessing after split
        # such as scaling 
        if after_split_preprocessing_fn != None:
            X_train, y_train, X_test, y_test = after_split_preprocessing_fn(X_train, 
                                                            y_train, X_test, y_test, **kwargs)

        clf.fit(X_train, y_train)

        y_train_preds = clf.predict(X_train)
        y_test_preds  = clf.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_preds)
        test_accuracy  = accuracy_score(y_test, y_test_preds)

        train_f1 = f1_score(y_train, y_train_preds, average='macro')
        test_f1  = f1_score(y_test, y_test_preds, average='macro')

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        train_f1_scores.append(train_f1)
        test_f1_scores.append(test_f1)

    # Getting each score's mean and standard deviation
    train_accuracies_mean = np.array(train_accuracies).mean()
    train_accuracies_std  = np.array(train_accuracies).std()

    test_accuracies_mean  = np.array(test_accuracies).mean()
    test_accuracies_std   = np.array(test_accuracies).std()

    train_f1_mean         = np.array(train_f1_scores).mean()
    train_f1_std          = np.array(train_f1_scores).std()

    test_f1_mean          = np.array(test_f1_scores).mean()
    test_f1_std           = np.array(test_f1_scores).std()

    return_tuple = (train_accuracies_mean, train_accuracies_std, test_accuracies_mean, 
                        test_accuracies_std, train_f1_mean, train_f1_std, 
                        test_f1_mean, test_f1_std)
    
    return return_tuple
