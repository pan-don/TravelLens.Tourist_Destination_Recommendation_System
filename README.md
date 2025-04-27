# TravelLens

```
app/                    # package aplikasi
├── __init__.py
├── main/               # blueprint utama
│   ├── __init__.py
│   └── routes.py   
├── models/             # package model
│   ├── __init__.py
│   ├── model.py        # file model
│   └── inference.py    # untuk inferensi/predik input
├── preprocessing/      # package preprocessing
│   ├── __init__.py
│   ├── tokenize.py
│   └── pipeline.py     # gabungan metode preprocessing
├── services/           # buat baca link image
│   └── image_service.py
├── static/             # nyimpen file static
└── templates/          # HTML templates (index.html, result.html)

data/                   # nyimpen data-data
├── dataset.csv
├── tfidf_matrix.pkl
├── tfidf_vectorizer.pkl
└── loader.py # buat load data-data tersebut

scripts/                # CLI / training scripts
├── training/
│   └── train.py
└── testing/
    └── train.py

requirements.txt
run.py                  # entrypoint
README.md
```