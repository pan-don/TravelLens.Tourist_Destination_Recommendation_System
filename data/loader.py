import pandas as pd
from gensim.models import Word2Vec

def get_dataset(file_path: str="data/tourism_with_image_clean.csv") -> pd.DataFrame:
    """
    Fungsi untuk memuat dataset yang telah dibersihkan dari file CSV
    dan mengembalikannya sebagai DataFrame.
    
    Parameters
    ----------
    file_path : str
        Path ke file CSV yang berisi dataset destinasi wisata.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame yang berisi data destinasi wisata.
    """
    df = pd.read_csv(file_path)
    return df

def get_pretrained_model(file_path: str="data/cbf_model.model") -> Word2Vec:
    """
    Fungsi untuk memuat model Word2Vec yang sudah dilatih sebelumnya
    dan mengembalikannya dalam format `gensim.models.Word2Vec`.
    
    Parameters
    ----------
    file_path : str
        Path ke file model Word2Vec yang sudah dilatih sebelumnya.
    
    Returns
    -------
    model : gensim.models.Word2Vec
        Model Word2Vec yang sudah dilatih sebelumnya.
    """
    model = Word2Vec.load(file_path)
    return model