import numpy as np
import pandas as pd
from typing import Literal
from gensim.models import Word2Vec
from data import get_dataset, get_pretrained_model
from app.preprocessing import pipeline, get_average_embeddings, get_filtered_df



def recommendation_system(description: str, 
                          category: Literal['Budaya', 'Taman Hiburan', 'Cagar Alam', 'Bahari', 'Pusat Perbelanjaan', 'Tempat Ibadah'], 
                          city: Literal['Jakarta', 'Yogyakarta', 'Bandung', 'Semarang', 'Surabaya'], 
                          df: pd.DataFrame=get_dataset(), 
                          model: Word2Vec=get_pretrained_model(), 
                          top_n: int=3) -> pd.DataFrame:
    """
    Fungsi untuk memberikan rekomendasi destinasi wisata berdasarkan deskripsi, kategori, dan kota.

    Parameters
    ----------
    description : str
        Deskripsi destinasi wisata yang ingin direkomendasikan.
    category : str
        Kategori yang dipilih oleh pengguna.
    city : str
        Kota yang dipilih oleh pengguna.
    df : pd.DataFrame
        Data destinasi wisata.
    model : Word2Vec
        Model Word2Vec yang telah dilatih sebelumnya.
    top_n : int
        Jumlah rekomendasi yang ingin ditampilkan.

    Returns
    -------
    pd.DataFrame
        DataFrame yang berisi top_n rekomendasi destinasi wisata.
    """
    
    # Filter data sesuai kategori dan kota
    df_filtered = get_filtered_df(df, category, city)
    if df_filtered.empty:
        print("No data found for the given category and city.")
        return df_filtered

    # preprocessing input data deskripsi
    cleaned_text = pipeline(description)
    
    # merepresentasikan input dan filtered data kedalam bentuk vektor supaya dapat dibaca model
    input_embeddings = get_average_embeddings(cleaned_text, model)
    filtered_embeddings = get_average_embeddings(df_filtered['Cleaned'], model)

    # Hitung similarity antara input dan filtered data
    similarity = model.wv.cosine_similarities(input_embeddings.flatten(), filtered_embeddings)

    # Ambil top-n berdasarkan similarity tertinggi
    top_similarity_idx = np.argsort(similarity)[::-1][:top_n]
    # Ambil baris dari df_filtered sesuai indeks top similarity
    df_recommendations = df_filtered.iloc[top_similarity_idx].copy()
    return df_recommendations