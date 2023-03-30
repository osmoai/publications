import numpy as np
import pandas as pd
from typing import Iterable, Union

from qian_et_al_2023.src import base


def load_gnn_rf_predictions(mol_codes):
    """Load GNN and RF ratings for a set of RedJade codes.
    
    Args:
        mol_codes: Molecules for which to retrieve predictions
    """
    files = {
        'RF': base.DATA_PATH / 'Data S4.csv',
        'GNN': base.DATA_PATH / 'Data S5.csv',
    }
    return {
        model_name: pd.read_csv(prediction_file
            ).set_index('RedJade Code'
            ).loc[mol_codes]
        for model_name, prediction_file in files.items()
    }


def load_extended_predictions(mol_codes):
    files = {
        'GNN': base.DATA_PATH / 'Data S5.csv',
        'RF_cFP': base.DATA_PATH / 'Data S4.csv',
        'RF_Mordred': base.DATA_PATH / 'standardized_rf_Mordred.csv',
        'KNN_cFP': base.DATA_PATH / 'standardized_knn_cFP.csv',
        'KNN_Mordred': base.DATA_PATH / 'standardized_knn_Mordred.csv',
    }
    return {
        model_name: pd.read_csv(prediction_file
            ).set_index('RedJade Code'
            ).loc[mol_codes]
        for model_name, prediction_file in files.items()
    }


def load_screening_ratings():
    return pd.read_csv(base.DATA_PATH / 'Data S2.csv')


def load_ratings(filter:Union[bool, Iterable]=True):
    """Load a dataframe with rater data and molecule metadata."""
    odor_key = pd.read_csv(base.DATA_PATH / 'Data S1.csv')
    
    ratings = pd.read_csv(base.DATA_PATH  / 'Data S3.csv')
    odor_key_fixed = odor_key[['RedJade Code', 'GC-O result',
                               'GC-O contaminant, if identified',
                               'Disqualification reason',
                               'Selection reason',
                              ]].rename(
            columns={'Odor Name': 'chemical_name', 'Kit': 'Round'})
    ratings = pd.merge(odor_key_fixed, ratings, how='inner',
                       right_on='SampleIdentifier', left_on='RedJade Code')

    if filter is True:
        ratings = ratings[ratings['Disqualification reason'].isna()]
    elif filter:  # filter is a list of disqualification reasons
        ratings = ratings[~ratings['Disqualification reason'].isin(filter)]
    return ratings


def get_clean(filter=True):
    # Load the data
    dfs = {}
    humans = load_ratings(filter=filter)
    mol_codes = humans['RedJade Code'].unique()
    models = load_gnn_rf_predictions(mol_codes)

    # Select the humans that rated at least some of these SMILES
    subjects = humans['SubjectCode'].unique()
    
    panel = humans.groupby('RedJade Code').mean().loc[mol_codes, base.MONELL_CLASS_LIST]
    
    return models, humans, panel, subjects


def load_embeddings():
    emb = np.load(base.DATA_PATH / 'gnn_embeddings.npz', allow_pickle=True)
    emb = pd.DataFrame(emb['prediction'], index=emb['smiles'])
    # Remove duplicate entries
    emb = emb[~emb.index.duplicated()]
    return emb


def load_redjade_smiles():
    return pd.read_csv(base.DATA_PATH / "Data S7.csv")