import pandas as pd

MODEL_FILE = './data/embeddings.csv'
MODEL_DF = pd.read_csv(MODEL_FILE, index_col=0)

def get_embs(smiles):
    return MODEL_DF.loc[smiles].values