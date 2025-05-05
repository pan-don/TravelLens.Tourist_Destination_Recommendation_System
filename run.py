from scripts.execute_notebook import run_notebook
from sklearn.feature_extraction.text import TfidfVectorizer
from data import get_vectorizer

if __name__ == "__main__":
    
    run_notebook('scripts/notebook/cleaning_dataset.ipynb')