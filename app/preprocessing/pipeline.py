from app.preprocessing.text_processing import Preprocessing

def Pipeline(input_text: str, inference: bool=True) -> str:
    clean_text = Preprocessing(input_text=input_text, inference=inference)
    result = clean_text.pipeline()
    return result