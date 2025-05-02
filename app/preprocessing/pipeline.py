from app.preprocessing.text_preprocessing import TextProcessing

def Pipeline(input_text: str, language: str='indonesian', tfidf_step: bool=True) -> str:
    clean_text = TextProcessing(input_text, language=language, tfidf_step=tfidf_step)
    result = clean_text.pipeline()
    return result