from data.data_loader import get_dataset
from app.models import ContentBasedFilteringModel
from app.preprocessing import Pipeline

df = get_dataset()

# fungsi untuk dekode data category
def decode_category(idx):
    category_dict = {i: cat for i, cat in enumerate(sorted(df['Category'].unique().tolist()))}
    result = [values for key, values in category_dict.items() if idx == key][0]
    return result

# fungsi untuk dekode data city
def decode_city(idx):
    city_dict = {i: cit for i, cit in enumerate(sorted(df['City'].unique().tolist()))}
    result = [values for key, values in city_dict.items() if idx == key][0]
    return result

def recommender_system(input_description: str, input_category: int, input_city: int, top_num: int=3):
    """
    Sistem rekomendasi wisata dengan 3 input:
    - `input_description` -> datatype: str
      masukan penjelasan tempat wisata yang ingin dikunjungi
      
    - `input_category` -> datatype: int
      masukan kategori tempat wisata yang ingin dikunjungi
      1. Bahari
      2. Budaya
      3. Cagar Alam
      4. Pusat Perbelanjaan
      5. Taman Hiburan
      6. Tempat Ibadah
    
    - `input_city` -> datatype: int
    masukan kota tempat wisata yang ingin dikunjungi
    1. Bandung
    2. Jakarta
    3. Semarang
    4. Surabaya
    5. Yogyakarta
    
    - `top_num` -> datatype int
    jumlah rekomendasi wisata yang ingin ditampilkan
    """
    
    description = Pipeline(input_description)
    category = decode_category(input_category - 1)
    city = decode_city(input_city - 1)
    
    model = ContentBasedFilteringModel(top_num=top_num)
    result_dict = model.predict(input_text=description, input_catg=category, input_city=city)
    return result_dict
    