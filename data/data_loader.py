# set pythonpath ke root projek
import pickle
import pandas as pd

def get_dataset():
    df = pd.read_csv("data/tourism_with_id.csv")
    return df

def get_vocab():
    with open('data/vocab_descriptions.pkl', 'rb') as f:
        vocab = pickle.load(f)
        return vocab