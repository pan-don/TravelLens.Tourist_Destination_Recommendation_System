import re
import torch
from string import punctuation
from nltk.tokenize import word_tokenize
<<<<<<< Updated upstream
from data import get_vocab
=======
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

>>>>>>> Stashed changes

class Preprocessing():
    def __init__(self, input_text: str, inference: bool):
        self.text = input_text
        self.inference_mode = inference
        
    def cleaning_text(self, input_text: str):
        text_lower = input_text.lower()
        clean = re.sub(f"[{re.escape(punctuation)}]", '', text_lower)
        return clean

    def tokenization(self, input_text: str):
        tokens = word_tokenize(input_text)
        return tokens
    
    def encoding_text(self, input_text: str, len_max: int=10):
        vocab = get_vocab()
        tokens = self.tokenization(input_text)
        idx = [vocab.get(word, _) for word in tokens] # type: ignore
        padding = idx                                                                       [:len_max]+[0] * (len_max - len(idx))
        tensor = torch.tensor(padding)
        return tensor
    
    def text_pipeline(self):
        input_text = self.text
        clean_text = self.cleaning_text(input_text)
        if not self.inference_mode:
            tokens = self.tokenization(clean_text)
            return tokens
        encoded_text = self.encoding_text(tokens, len_max=10)
        return encoded_text
