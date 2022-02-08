import itertools
import sys
import time

sys.path.append('../Utils/')

import numpy  as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from train_utils import ClassifierGridSearch
from train_utils import RepeatedKFoldCrossValidation



# clf params is just a "kwargs" dict for each clf
# Also, this will create a bar plot
#TODO: SHOW IN THESE PLOTS THE TRAIN AND PREDICT TIME PER SAMPLE
def evaluateBestClassifiers(clf_refs, clf_params, X, y, X_val, y_val, k=10, repeats=2, 
                        after_split_preprocessing_fn=None, kwargs={}):
    
    new_X = pd.DataFrame(X)
    new_y = pd.Series(y)

    new_X_val = pd.DataFrame(X_val)
    new_y_val = pd.Series(y_val)

    confusion_matrix_labels = y_val.unique()
    confusion_matrices      = []

    mean_train_acc_list = []
    train_acc_std_list  = []
    mean_test_acc_list  = []
    test_acc_std_list   = []
    validation_acc_list = []

    mean_train_f1_list  = []
    train_f1_std_list   = []
    mean_test_f1_list   = []
    test_f1_std_list    = []
    validation_f1_list  = []

    # Keeping the train and prediction times per sample in order to show
    # the performance of chosen classifiers, in order to see
    # how they'll perform in demo, thus, choose the best one
    # taking into account the other metrics
    train_times_ms      = []
    prediction_times_ms = []

    for clf_ref, clf_kwargs in zip(clf_refs, clf_params):
        clf = clf_ref(**clf_kwargs)

        result = RepeatedKFoldCrossValidation(k, repeats, clf, X, y, 
                                        after_split_preprocessing_fn, kwargs)
        
        train_mean_acc, train_acc_std, test_mean_acc, test_acc_std,\
        train_mean_f1, train_f1_std, test_mean_f1, test_f1_std       = result

        # Getting Validation Accuracy
        if after_split_preprocessing_fn != None:
            curr_X, curr_y, curr_X_val, curr_y_val = after_split_preprocessing_fn(new_X,
                                                new_y, new_X_val, new_y_val, **kwargs)

        time_train_ms = time.time()

        clf.fit(curr_X, curr_y)
        
        time_train_ms = ((time.time() - time_train_ms) / len(curr_X)) * 1000

        predict_time_ms = time.time()

        y_val_preds    = clf.predict(curr_X_val)
        
        predict_time_ms = ((time.time() - predict_time_ms) / len(curr_X_val)) * 1000

        validation_acc = accuracy_score(curr_y_val, y_val_preds)
        validation_f1  = f1_score(curr_y_val, y_val_preds, average='macro')

        conf_matrix    = confusion_matrix(curr_y_val, y_val_preds, labels=confusion_matrix_labels)

        mean_train_acc_list.append(train_mean_acc)
        train_acc_std_list.append(train_acc_std)
        mean_test_acc_list.append(test_mean_acc)
        test_acc_std_list.append(test_acc_std)
        validation_acc_list.append(validation_acc)

        mean_train_f1_list.append(train_mean_f1)
        train_f1_std_list.append(train_f1_std)
        mean_test_f1_list.append(test_mean_f1)
        test_f1_std_list.append(test_f1_std)
        validation_f1_list.append(validation_f1)

        train_times_ms.append(round(time_train_ms, 4))
        prediction_times_ms.append(round(predict_time_ms, 4))

        confusion_matrices.append(conf_matrix.tolist())
    
    # Plotting the results among training, testing and validation accuracy
    # and F1 score
    x_ticks = [clf.__name__ + '(' + str(clf_kwargs) + ')' 
                for clf, clf_kwargs in zip(clf_refs, clf_params)]

    fig = make_subplots(rows=2, cols=1)#go.Figure()

    metrics             = ['Accuracy',          'F1-Score']
    train_y_ticks_list  = [mean_train_acc_list, mean_train_f1_list]
    train_errors_y_list = [train_acc_std_list,  train_f1_std_list]
    test_y_ticks_list   = [mean_test_acc_list,  mean_test_f1_list]
    test_errors_y_list  = [test_acc_std_list,   test_f1_std_list]
    val_y_ticks_list    = [validation_acc_list, validation_f1_list]

    for elem in zip(range(1, 3), metrics, train_y_ticks_list, train_errors_y_list, 
                    test_y_ticks_list, test_errors_y_list, val_y_ticks_list):

        row_num, metric, train_y_ticks, train_errors_y,\
        test_y_ticks, test_errors_y, val_y_ticks         = elem

        fig.add_trace(go.Bar(name='Training ' + metric, x=x_ticks, y=train_y_ticks,
                    error_y={
                        'type': 'data', 
                        'array': [2 * error for error in train_errors_y]
                    }), row=row_num, col=1)

        fig.add_trace(go.Bar(name='Testing ' + metric, x=x_ticks, y=test_y_ticks,
                    error_y={
                        'type': 'data', 
                        'array': [2 * error for error in test_errors_y]
                    }), row=row_num, col=1)

        fig.add_trace(go.Bar(name='Validation ' + metric, x=x_ticks, y=val_y_ticks),
                        row=row_num, col=1)

    fig.update_layout(title='Best Classifiers training, testing and validation Accuracy and F1 Score with 95\% confidence intervals' +
                            'with Repeated-K-Fold with k=' + str(k) + ' and ' + str(repeats) + ' repeats.')

    fig.show()

    # Plotting the confusion matrix of each classifier in validation data
    cm_titles = [clf_ref.__name__ + '(' + str(clf_kwargs) + ')'
                    for clf_ref, clf_kwargs in zip(clf_refs, clf_params)]

    fig = make_subplots(rows=1, cols=len(clf_refs), subplot_titles=cm_titles)

    for i, conf_matrix in zip(range(1, len(clf_refs) + 1),confusion_matrices):
        show_legend = True if i == 1 else False

        fig.add_trace(go.Heatmap(z=conf_matrix, text=conf_matrix, x=confusion_matrix_labels,
                                y=confusion_matrix_labels, texttemplate='%{text}', textfont={"size":20},
                                legendgroup=1, showlegend=show_legend), row=1, col=i)

    fig.update_annotations(font_size=14)

    fig.show()

    # Plotting the time performance of each classifier
    fig = go.Figure()

    fig.add_trace(go.Bar(name='Training time per sample(ms)',   x=x_ticks, y=train_times_ms))

    fig.add_trace(go.Bar(name='Prediction time per sample(ms)', x=x_ticks, y=prediction_times_ms))

    fig.update_layout(title='Best Classifiers Training and Validation Performance.')
    
    fig.show()

    return



def evaluateClassifiersTraining(clf_refs, clf_params_list, X, y, k=10, repeats=2, 
                                after_split_preprocessing_fn=None):

    for clf_ref, clf_params in zip(clf_refs, clf_params_list):
        trainClassifierAndPlotResults(clf_ref, clf_params, X, y, k=k, repeats=repeats,
                                    after_split_preprocessing_fn=after_split_preprocessing_fn)

    return



# Training the classifier through the 4 types of datasets
# that I have, for each of its hyperparameters and plotting
# the results
def trainClassifierAndPlotResults(clf_ref, clf_params, X, y, k=10, repeats=2, 
                                after_split_preprocessing_fn=None):
    results = _train_clf(clf_ref, clf_params, X, y, k=k, repeats=repeats, 
                        after_split_preprocessing_fn=after_split_preprocessing_fn)
    
    _plot_results(clf_ref.__name__, results['original_dataset'],
                results['doubled_dataset'], results['diff_dataset'],
                results['doubled_diff_dataset'])

    return



def _train_clf(clf_ref, clf_params, X, y, k=10, repeats=2, 
            after_split_preprocessing_fn=None):
    
    result_dict = {}
    
    result_dict['original_dataset'] = ClassifierGridSearch(clf_ref, clf_params,
                                    X, y, k, repeats, after_split_preprocessing_fn)
    
    result_dict['doubled_dataset']  = ClassifierGridSearch(clf_ref, clf_params,
                                        X, y, k, repeats, after_split_preprocessing_fn,
                                        {'to_double': True})

    result_dict['diff_dataset']     = ClassifierGridSearch(clf_ref, clf_params,
                                    X, y, k, repeats, after_split_preprocessing_fn,
                                    {'to_diff': True})

    result_dict['doubled_diff_dataset'] = ClassifierGridSearch(clf_ref, clf_params,
                                            X, y, k, repeats, after_split_preprocessing_fn,
                                            {'to_double': True, 'to_diff': True})

    return result_dict



def _plot_results(clf_name, og_dataset_results, double_dataset_results,
                            diff_dataset_results, double_diff_dataset_results):
    titles = [
        'Accuracy and F1-Score Margin on Original Dataset',
        'Accuracy and F1-Score Margin on Double Dataset',
        'Accuracy and F1-Score Margin on Difference Dataset',
        'Accuracy and F1-Score Margin on Double Difference Dataset'
    ]

    x_ticks = [str(x) for x in og_dataset_results['clf_parameters']]

    fig = make_subplots(rows=2, cols=2, subplot_titles=titles)

    positions = itertools.product(range(1, 3), range(1, 3))

    datasets  = [ og_dataset_results, double_dataset_results,
                diff_dataset_results, double_diff_dataset_results ]

    for pos, curr_dataset in zip(positions, datasets):
        curr_row, curr_col = pos
        show_legend        = False

        if curr_row == 1 and curr_col == 1:
            show_legend = True
        

        fig.add_trace(
            go.Scatter(x=x_ticks, y=curr_dataset['train_mean_accuracy'], 
                        marker=go.scatter.Marker(color='red'),
                        error_y={
                            'type':    'data',
                            'array':   [2 * std for std in curr_dataset['train_accuracy_std']],
                            'visible': True
                        },
                        legendgroup='train', name='Training Accuracy', showlegend=show_legend),

            row=curr_row, col=curr_col
        )

        fig.add_trace(
            go.Scatter(x=x_ticks, y=curr_dataset['test_mean_accuracy'], 
                        marker=go.scatter.Marker(color='blue'),
                        error_y={
                            'type':    'data',
                            'array':   [2 * std for std in curr_dataset['test_accuracy_std']],
                            'visible': True
                        },
                        legendgroup='test', name='Testing Accuracy', showlegend=show_legend),

            row=curr_row, col=curr_col
        )

        fig.add_trace(
            go.Scatter(x=x_ticks, y=curr_dataset['train_mean_f1_score'], 
                        marker=go.scatter.Marker(color='green'),
                        error_y={
                            'type':    'data',
                            'array':   [2 * std for std in curr_dataset['train_f1_score_std']],
                            'visible': True
                        },
                        legendgroup='train', name='Training F1-Score', showlegend=show_legend),

            row=curr_row, col=curr_col
        )

        fig.add_trace(
            go.Scatter(x=x_ticks, y=curr_dataset['test_mean_f1_score'], 
                        marker=go.scatter.Marker(color='goldenrod'),
                        error_y={
                            'type':    'data',
                            'array':   [2 * std for std in curr_dataset['test_f1_score_std']],
                            'visible': True
                        },
                        legendgroup='test', name='Testing F1-Score', showlegend=show_legend),

            row=curr_row, col=curr_col
        )


    fig.update_yaxes(title='Score',   fixedrange=True, range=[-0.2, 1.2])
    fig.update_xaxes(title='Parameters')

    fig.update_layout(title=clf_name + '\'s Train and Test, Accuracy and F1-Score on different Datasets')

    fig.show()

    return