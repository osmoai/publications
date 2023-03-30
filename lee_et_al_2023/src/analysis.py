import itertools
import random

import numpy as np
import pandas as pd

from lee_et_al_2023.src import base

def center_model(model):
    """Center a model."""
    new_model = model.copy()
    all_ratings = new_model[base.MONELL_CLASS_LIST].values
    bias = np.mean(all_ratings, axis=0)
    new_model[base.MONELL_CLASS_LIST] = all_ratings - bias
    return new_model

def fast_process(humans, models, f=None, subjects=None, center_fn=center_model,
             descriptors=base.MONELL_CLASS_LIST, subgroup_size=1, axis=0):
    # humans: Either a dataframe of all the data OR a dict of individual human data in the same shape and style as the model dataframes.
    # models: A dict of individual model predictions.
    # f: A transformation to apply before computing correlation
    if f is None:
        f = lambda x: x
    
    if axis == 0:
        corr_fn = lambda df1, df2: f(df1).T.corrwith(f(df2).T)
    elif axis == 1:
        corr_fn = lambda df1, df2: f(df1).corrwith(f(df2))
    if subjects is None:
        if isinstance(humans, pd.DataFrame):
            subjects = humans['SubjectCode'].unique().tolist()
        else:
            subjects = list(humans.keys())
    subject_groups = list(itertools.combinations(subjects, subgroup_size))
    random.shuffle(subject_groups)
    subject_groups = subject_groups[:100] # For computational reasons...
    columns = list(models.keys()) + ['Human']
    corrs = []
    for subj_group in subject_groups:
        for rep in [1, 2]:
            # Slice out one subject and a subpanel of the remaining subjects
            human = humans[humans['SubjectCode'].isin(subj_group) & (humans['Rep'] == rep)
                          ].groupby('RedJade Code')[descriptors].mean()
            panel = humans[~humans['SubjectCode'].isin(subj_group)
                          ].groupby('RedJade Code')[descriptors].mean()
            # Align indices and select the subset of molecules that this human rated 
            panel = panel.loc[human.index]
            human[:] = center_fn(human)
            panel[:] = center_fn(panel)
            _rows = pd.DataFrame({
                'SubjectCode' : ', '.join(subj_group),
                'Rep': rep,
                'Human': corr_fn(human, panel)
            })
            # Compute correlation for each model with the same subpanel
            for model_name, model in models.items():
                model_ = model.loc[human.index][descriptors]
                model_[:] = center_fn(model_)
                _rows[model_name] = corr_fn(model_, panel)
            corrs.append(_rows)
    corrs = pd.concat(corrs)
    corrs = corrs.reset_index().set_index(['RedJade Code' if axis == 0 else 'index', 'SubjectCode', 'Rep'])
    return corrs


def model_performance(corrs, model_name):
    """
    Reports the performance of `model_name` in matrix `corrs` along
    `axis`, which determines aggregation across either descriptors or
    across molecules.
    
    Params:
        corrs: The output of `fast_corrs`.
        model_name: e.g. 'GNN'
    Returns:
        A pd.Series indexed by molecules or descriptors
        and containing win rates.
    """
    n_wins = (corrs[model_name] > corrs['Human']).groupby('RedJade Code').sum()
    n_ties = (corrs[model_name] == corrs['Human']).groupby('RedJade Code').sum()
    n_losses = (corrs[model_name] < corrs['Human']).groupby('RedJade Code').sum()
    return (n_wins + 0.5*n_ties) / (n_wins + n_ties + n_losses)


def shuffle_df(df, shuffle):
    df_contents = df.to_numpy()
    if shuffle == 'descriptors':
        # Generate an independent descriptor shuffle for each molecule
        shuffle_array = np.argsort(np.random.random(df_contents.shape), axis=1)
        df_contents = df_contents[np.arange(df_contents.shape[0])[:, np.newaxis], shuffle_array]
    elif shuffle == 'molecules':
        df_contents = df_contents[np.random.permutation(df_contents.shape[0])]
    else:
        raise ValueError('Unsupported shuffle technique: {}'.format(shuffle))
    return pd.DataFrame(df_contents, index=df.index, columns=df.columns)
