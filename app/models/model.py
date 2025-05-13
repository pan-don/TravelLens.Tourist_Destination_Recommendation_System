import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data.data_loader import get_dataset, get_vectorizer, get_tfidf_matrix
from typing import Callable, Dict, Any, List

class BaseRecommender:
    """Kelas dasar untuk semua recommenders (prinsip: inheritance, polymorphism)."""
    def predict(self, *args, **kwargs):
        """Metode abstract predict yang akan di-overridde oleh subclass."""
        raise NotImplementedError("Subclasses should implement this method.")

class ContentBasedFilteringModel(BaseRecommender):
    """
    Content-based filtering recommender dengan prinsip functional dan OOP.
    Metode rekomendasi menggunakan informasi konten dari item untuk memberikan rekomendasi.
    """
    def __init__(
        self,
        top_num: int = 3,
        vectorizer: TfidfVectorizer = get_vectorizer(),
        tfidf_matrix: np.ndarray = get_tfidf_matrix(),
        input_dataset: pd.DataFrame = get_dataset()
    ):
        self._top_num = top_num
        self._vectorizer = vectorizer
        self._tfidf_matrix = tfidf_matrix
        self._dataset = input_dataset

    @property
    def top_num(self) -> int:
        return self._top_num

    @property
    def vectorizer(self) -> TfidfVectorizer:
        return self._vectorizer

    @property
    def tfidf_matrix(self) -> np.ndarray:
        return self._tfidf_matrix

    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset

    def _filter_indices(
        self,
        filter_fn: Callable[[pd.DataFrame, int], bool]
    ) -> List[int]:
        """
        Fungsi untuk memfilter indeks dataset (prinsip higher-order function).
        """
        return [i for i in range(len(self._dataset)) if filter_fn(self._dataset, i)]

    def filter_indices_by_category_and_city(
        self,
        input_catg: str,
        input_city: str
    ) -> List[int]:
        """
        Filter indeks dataset berdasarkan kategori dan kota.
        Mengembalikan list indeks sesuai kriteria.
        """
        return [
            i for i in range(len(self._dataset))
            if self._dataset['Category'].iloc[i] == input_catg and self._dataset['City'].iloc[i] == input_city
        ]

    def predict(
        self,
        input_text: str,
        input_catg: str,
        input_city: str
    ) -> Dict[str, List[Any]]:
        """
        Merekomendasikan tempat berdasarkan input teks, kategori, dan kota.
        Mengembalikan dictionary dari recommendations.
        """
        input_tfidf_matrix = self._vectorizer.transform([input_text])
        cosims = cosine_similarity(input_tfidf_matrix, self._tfidf_matrix)[0]
        filtered_data = self.filter_indices_by_category_and_city(input_catg, input_city)

        if filtered_data:
            filtered_cosims = cosims[filtered_data]
            sorted_order = np.argsort(filtered_cosims)[::-1][:self._top_num]
            recommend_idx = [filtered_data[i] for i in sorted_order]

            recommendation_dict = {
                "Name": [self._dataset['Place_Name'].iloc[i] for i in recommend_idx],
                "Category": [self._dataset['Category'].iloc[i] for i in recommend_idx],
                "City": [self._dataset['City'].iloc[i] for i in recommend_idx],
                "Link Image": [self._dataset['Link_Image'].iloc[i] for i in recommend_idx]
            }
            return recommendation_dict
        else:
            return {"Name": [], "Category": [], "City": [], "Link Image": []}