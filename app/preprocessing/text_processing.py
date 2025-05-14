import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)


class Preprocessing():
    def __init__(self, input_text: str, language: str='indonesian'):
        self.text = input_text
        self.lang = language
        
    def lower_case(self, input_text: str):
        return input_text.lower()
    
    def cleaning_text(self, input_text: str) -> str:
        clean_text = [char for char in input_text if char not in punctuation]
        return ''.join(clean_text)

    def remove_stop_words(self, input_text: str, lang: str) -> str:
        tokens = word_tokenize(input_text)
        stops = set(stopwords.words(lang))
        clean_text = [word for word in tokens if not word in stops]
        clean_text = " ".join(clean_text)
        return clean_text
    
    def text_pipeline(self):
        text1 = self.lower_case(self.text)
        text2 = self.cleaning_text(text1)
        text3 = self.remove_stop_words(text2, self.lang)
        return text3



def pipeline(input_text: str, lang: str='indonesian') -> str:
    clean_text = Preprocessing(input_text=input_text, language=lang)
    result = clean_text.text_pipeline()
    return result