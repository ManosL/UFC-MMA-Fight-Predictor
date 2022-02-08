import sys

sys.path.append('../Utils')

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from preprocessing_utils import df_drop_columns



def dim_reduction_eda(X_og, y_og, X_double, y_double, 
                    X_diff, y_diff, X_double_diff, y_double_diff):

    # Doing Dimensionality Reduction via tSNE
    tsne = TSNE(n_components=2)
    
    print('Start tSNE')

    X_og_reduced          = tsne.fit_transform(X_og.values)
    X_og_reduced          = pd.DataFrame(X_og_reduced)

    print('Finished for Original Dataset')
    
    X_double_reduced      = tsne.fit_transform(X_double.values)
    X_double_reduced      = pd.DataFrame(X_double_reduced)
    
    print('Finished for Double Dataset')
    
    X_diff_reduced        = tsne.fit_transform(X_diff.values)
    X_diff_reduced        = pd.DataFrame(X_diff_reduced)
    
    print('Finished for Difference Dataset')
    
    X_double_diff_reduced = tsne.fit_transform(X_double_diff.values)
    X_double_diff_reduced = pd.DataFrame(X_double_diff_reduced)
    
    print('Finished for Double Difference Dataset')
    
    # Plotting Dimensionality Reduction results of each dataset
    titles = [
        'tSNE Results on original Dataset',
        'tSNE Results on Dataset with doubled rows via reversing also Fighter\'s features in each fight(and also the labels)',
        'tSNE Results where we taken the difference between each fighter\'s numerical features',
        'tSNE Results on a doubled dataset which we\n also took stats\' difference'
    ]
    
    labels       = ['win', 'lose', 'draw']
    label_colors = ['red', 'green', 'blue']

    fig = make_subplots(rows=2, cols=2, subplot_titles=titles)

    for label, color in zip(labels, label_colors):
        X_og_label = X_og_reduced[y_og == label].values
        fig.add_trace(
            go.Scatter(x=X_og_label[:, 0], y=X_og_label[:, 1], mode="markers", 
                        marker=go.scatter.Marker(color=color),
                        legendgroup=label, name=label, showlegend=True),

            row=1, col=1
        )

    for label, color in zip(labels, label_colors):
        X_double_label = X_double_reduced[y_double == label].values
        fig.add_trace(
            go.Scatter(x=X_double_label[:, 0], y=X_double_label[:, 1], mode="markers", 
                        marker=go.scatter.Marker(color=color),
                        legendgroup=label, name=label, showlegend=False),

            row=1, col=2
        )

    for label, color in zip(labels, label_colors):
        X_diff_label = X_diff_reduced[y_diff == label].values
        fig.add_trace(
            go.Scatter(x=X_diff_label[:, 0], y=X_diff_label[:, 1], mode="markers",
                        marker=go.scatter.Marker(color=color),
                        legendgroup=label, name=label, showlegend=False),

            row=2, col=1
        )

    for label, color in zip(labels, label_colors):
        X_double_diff_label = X_double_diff_reduced[y_double_diff == label].values
        fig.add_trace(
            go.Scatter(x=X_double_diff_label[:, 0], y=X_double_diff_label[:, 1], mode="markers", 
                        marker=go.scatter.Marker(color=color),
                        legendgroup=label, name=label, showlegend=False),

            row=2, col=2
        )

    fig.show()
    return

def correlation_matrices_eda(X_og, X_double, X_diff, X_double_diff):
    # Dropping categorical features
    new_X_og = df_drop_columns(X_og, ['Gender', 'Weight_Class', 'Title_Fight',
                                    'Fight_Time_Format', 'Fighter_1_Stance',
                                    'Fighter_2_Stance'])

    new_X_double = df_drop_columns(X_double, ['Gender', 'Weight_Class', 'Title_Fight',
                                    'Fight_Time_Format', 'Fighter_1_Stance',
                                    'Fighter_2_Stance'])

    new_X_diff = df_drop_columns(X_diff, ['Gender', 'Weight_Class', 'Title_Fight',
                                    'Fight_Time_Format', 'Fighter_1_Stance',
                                    'Fighter_2_Stance'])

    new_X_double_diff = df_drop_columns(X_double_diff, ['Gender', 'Weight_Class', 'Title_Fight',
                                    'Fight_Time_Format', 'Fighter_1_Stance',
                                    'Fighter_2_Stance'])

    # Plotting the correlation matrixes
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].matshow(new_X_og.corr())
    axs[0, 0].set_title('Correlation matrix in Original Dataset')

    axs[0, 1].matshow(new_X_double.corr())
    axs[0, 1].set_title('Correlation matrix in Double Dataset')

    axs[1, 0].matshow(new_X_diff.corr())
    axs[1, 0].set_title('Correlation matrix in Difference Dataset')

    axs[1, 1].matshow(new_X_double_diff.corr())
    axs[1, 1].set_title('Correlation matrix in Double Difference Dataset')

    plt.show()
    return