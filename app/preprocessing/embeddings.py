import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

def get_average_embeddings(texts: list[str], 
                           model: Word2Vec) -> np.ndarray:
    """
    Fungsi untuk menghitung rata-rata embedding dari teks yang diberikan
    menggunakan model Word2Vec yang telah dilatih sebelumnya.
    
    Parameters
    ----------
    texts : list[str]
        Daftar teks yang ingin dihitung rata-rata embedding-nya.
    model : Word2Vec
        Model Word2Vec yang telah dilatih sebelumnya.
    
    Returns
    -------
    average embeddings : np.ndarray
        Matriks rata-rata embedding dari teks yang diberikan.
    """
    
    # Mengubah input menjadi list jika bertipe string untuk memastikan konsistensi dalam pemrosesan
    if isinstance(texts, str):
        texts = [texts]
    
    # inisialisasi model Word2Vec 
    word_vector = model.wv
    vector_size = word_vector.vector_size
    
    # Menyimpan rata-rata embedding untuk setiap teks
    avg_embeddings = []
    # Menghitung rata-rata embedding untuk setiap teks
    for text in texts:
        try:
            tokens = word_tokenize(text)
            valid_tokens = [token for token in tokens if token in word_vector]
            # Mengecek apakah token tersebut ada dalam model Word2Vec
            # Jika tidak ada, maka token tersebut diabaikan
            if valid_tokens:
                matrix = np.vstack([word_vector[w] for w in valid_tokens])
                avg_embeddings.append(matrix.mean(axis=0))
            else:
                avg_embeddings.append(np.zeros(vector_size))
        except Exception as e:
            raise ValueError(f"Error in calculating average embeddings: {e}")
    
    return np.vstack(avg_embeddings)