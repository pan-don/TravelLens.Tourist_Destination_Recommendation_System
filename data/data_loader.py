# set pythonpath ke root projek
import os
import pickle
import pandas as pd

def get_dataset():
    df = pd.read_csv("data/tourism_with_id.csv")
    return df
    
def get_vectorizer():
    with open('data/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        return vectorizer

def get_tfidf_matrix():
    with open('data/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
        return tfidf_matrix