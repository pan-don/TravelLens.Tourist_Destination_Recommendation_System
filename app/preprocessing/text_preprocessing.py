import nltk
from nltk.data import find
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from data.data_loader import get_vectorizer, get_tfidf_matrix

class TextProcessing():
    def __init__(self, input_text: str, language: str, tfidf_step: str):
        self.text = input_text
        self.lang = language
        self.tfidf = tfidf_step
        self.check_nltk_resource('./data/lemmatize/wordnet')
        self.check_nltk_resource('./data/corpora/stopwords')
        self.check_nltk_resource('./data/tokenizers/punkt')
    
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

    def lemmatization(self, text: str) -> str:
        tokens = self.tokenization(text)
        lemmatizer = WordNetLemmatizer()
        lemmatize_words = [lemmatizer.lemmatize(token) for token in tokens]
        lemmatize_text = ' '.join(lemmatize_words)
        return lemmatize_text

    def tfidf_vectorizer(self, text: str):
        text_vectorizer = get_vectorizer.transform(text)
        return text_vectorizer        
        
    def cosine_similarity_matrix(self, text: str) -> str:
        result = cosine_similarity(text, get_tfidf_matrix)[0]
        return result
    
    def pipeline(self):
        text = self.text
        stopwords_text = self.remove_stop_word(text, lang=self.lang)
        stemming_text = self.stemming(stopwords_text)
        clear_text = self.lemmatization(stemming_text)
        if not self.tfidf:
            return clear_text
        tfidf_text = self.tfidf_vectorizer(clear_text)
        cosim_text = self.cosine_similarity_matrix(tfidf_text)
        return cosim_text