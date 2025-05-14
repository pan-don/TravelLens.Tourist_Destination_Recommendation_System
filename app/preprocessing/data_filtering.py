import pandas as pd

def get_filtered_df(df: pd.DataFrame, 
                    input_category: str, 
                    input_city: str) -> pd.DataFrame:
    """
    Fungsi untuk memfilter DataFrame berdasarkan kategori dan kota yang diberikan
    dan mengembalikan DataFrame yang telah difilter.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang berisi data destinasi wisata.
    input_category : str
        Kategori yang dipilih oleh pengguna.
    input_city : str
        Kota yang dipilih oleh pengguna.
    
    Returns
    -------
    df_filtered : pd.DataFrame
        DataFrame yang telah difilter berdasarkan kategori dan kota.
    """
    
    # Memfilter DataFrame berdasarkan kategori dan kota    
    df_filtered = df[
        (df['Category'] == input_category) & 
        (df['City'] == input_city)
    ]
    return df_filtered