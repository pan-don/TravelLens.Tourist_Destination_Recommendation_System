# set pythonpath ke root projek
import pickle
import pandas as pd

def get_dataset():
    df = pd.read_csv("data/tourism_with_image.csv")
    return df

def get_tfidf_matrix():
    with open('data/tfidf_matrix.pkl', 'rb') as f:
        matrix = pickle.load(f)
        return matrix

def get_vectorizer():
    with open('data/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        return vectorizer