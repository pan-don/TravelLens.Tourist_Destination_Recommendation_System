import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data.data_loader import get_dataset, get_vectorizer, get_tfidf_matrix


class ContentBasedFilteringModel():
    def __init__(self, top_num: int=3, vectorizer: TfidfVectorizer=get_vectorizer(), tfidf_matrix: np.ndarray=get_tfidf_matrix(), input_dataset: pd.DataFrame=get_dataset()):
        self.top_num = top_num
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.dataset = input_dataset
    
    def predict(self, input_text: str, input_catg: str, input_city: str):
        input_tfidf_matrix = self.vectorizer.transform([input_text])
        cosims = cosine_similarity(input_tfidf_matrix, self.tfidf_matrix)[0]
        
        filtered_data = [
        i for i in range(len(self.dataset['Place_Name']))
        if self.dataset['Category'][i] == input_catg and
        self.dataset['City'][i] == input_city
        ]
        if filtered_data:
            filtered_cosims = cosims[filtered_data]
            sorted_order = np.argsort(filtered_cosims)[::-1][:self.top_num]
            recommend_idx = [filtered_data[i] for i in sorted_order]
            
            #     # menyimpan nama, kota, dan kategori destinasi wisata yang direkomendasikan
            recomendation_dict = {"Name": [], "City": [], "Category": [], "Link Image": []}
            for i in recommend_idx:
                recomendation_dict["Name"].append(self.dataset['Place_Name'].iloc[i])
                recomendation_dict["Category"].append(self.dataset['Category'].iloc[i])
                recomendation_dict["City"].append(self.dataset['City'].iloc[i])
                recomendation_dict["Link Image"].append(self.dataset['Link_Image'].iloc[i])
            return recomendation_dict
        else:
            return []