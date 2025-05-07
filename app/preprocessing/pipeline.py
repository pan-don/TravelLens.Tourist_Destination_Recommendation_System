from app.preprocessing.text_processing import Preprocessing

def Pipeline(input_text: str, language: str='indonesian') -> str:
    """
    pipeline untuk preprocessing data teks
    """
    
    clean_text = Preprocessing(input_text=input_text, lang=language)
    result = clean_text.pipeline()
    return result