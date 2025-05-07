from app.models import recommender_system

def halaman():
    """
    List City:
    1. Bandung
    2. Jakarta
    3. Semarang
    4. Surabaya
    5. Yogyakarta

    List Category:
    1. Bahari
    2. Budaya
    3. Cagar Alam
    4. Pusat Perbelanjaan
    5. Taman Hiburan
    6. Tempat Ibadah

    """
    
    # input user
    city = int(input("Input City name   : ")) # input berupa nomor urut dari list nama kota
    catg = int(input("Input Category    : ")) # input berupa nomor urut dari list kategori
    desc = str(input("Input Description : "))
    
    results = recommender_system(input_description=desc, input_category=catg, input_city=city)
    
    print(results)
    print("Result:\n")
    for i in range(len(results)):
        print(f"{i+1}.".ljust(3, " ") + f"Name        : {results['Name'][i-1]}")
        print(" "*3 + f"City        : {results['City'][i-1]}")
        print(" "*3 + f"Category    : {results['Category'][i-1]}")
        print(" "*3 + f"Link        : {results['Link Image'][i-1]}")
        print()