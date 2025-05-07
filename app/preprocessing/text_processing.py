import nltk
import numpy as np
from nltk.data import find
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from data.data_loader import get_vectorizer, get_tfidf_matrix


class Preprocessing():
    """
    Preprocessing data teks seperti normalisasi, tokenisasi, konversi teks ke vektor, dan pipeline seluruh preprocessing data.
    """

    def __init__(self, input_text: str, lang: str):
        self.text = input_text
        self.lang = lang
        self.check_nltk_resource('data/lemmatize/wordnet')
        self.check_nltk_resource('data/corpora/stopwords')
        self.check_nltk_resource('data/tokenizers/punkt')
    
    @staticmethod
    def check_nltk_resource(path) -> str:
        try:
            find(path)
        except:
            nltk.download(path.split('/')[-1])

    def lower_case(self, text: str) -> str:
        return text.lower()

    def tokenization(self, text: str) -> list[str]:
        return word_tokenize(text)
    
    def remove_stop_word(self, text: str, lang: str) -> str:
        stop_words = set(stopwords.words(lang))
        tokens = self.tokenization(text)
        clean_words = [
            token for token in tokens 
            if token.isalnum() and token not in stop_words
        ]
        clean_text = ' '.join(clean_words)
        return clean_text

    def stemming(self, text: str) -> str:
        tokens = self.tokenization(text)
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stem_words = [stemmer.stem(token) for token in tokens]
        stem_text = ' '.join(stem_words)
        return stem_text
    
    def pipeline(self):
        text = self.text
        stopwords_text = self.remove_stop_word(text, lang=self.lang)
        clean_text = self.stemming(stopwords_text)
        return clean_text