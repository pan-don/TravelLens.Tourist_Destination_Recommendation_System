import nltk
import logging
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Mengecek dan mengunduh resource NLTK yang diperlukan jika belum tersedia
def nltk_resources():
    resources = ['punkt', 'stopwords', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)

# inisialisasi resource NLTK
nltk_resources()



class Preprocessing():
    """
    Kelas untuk melakukan praproses teks deskripsi wisata seperti 
    normalisasi huruf, pembersihan tanda baca, dan penghapusan stopwords.

    Parameters
    ----------
    input_text : str
        Teks deskripsi yang ingin diproses.
    language : str, optional
        Bahasa stopwords yang digunakan (default 'indonesian').

    Attributes
    ----------
    text : str
        Teks input yang akan diproses.
    lang : str
        Bahasa untuk stopwords.
    """

    def __init__(self, input_text: str, language: str='indonesian'):
        self.text = input_text
        self.lang = language

    def lower_case(self, input_text: str) -> str:
        """
        Mengubah semua huruf menjadi huruf kecil.

        Parameters
        ----------
        input_text : str
            Teks input.

        Returns
        -------
        str
            Teks dalam huruf kecil.
        """
        return input_text.lower()

    def cleaning_text(self, input_text: str) -> str:
        """
        Menghapus semua tanda baca dari teks.

        Parameters
        ----------
        input_text : str
            Teks input.

        Returns
        -------
        str
            Teks tanpa tanda baca.
        """
        clean_text = [char for char in input_text if char not in punctuation]
        return "".join(clean_text)

    def remove_stop_words(self, input_text: str, lang: str) -> str:
        """
        Menghapus kata-kata umum (stopwords) dari teks.

        Parameters
        ----------
        input_text : str
            Teks input yang sudah dibersihkan.
        lang : str
            Bahasa stopwords.

        Returns
        -------
        str
            Teks setelah stopwords dihapus.
        """
        try:
            tokens = word_tokenize(input_text)
            stops = set(stopwords.words(lang))
            clean_text = [word for word in tokens if word not in stops]
            return " ".join(clean_text)
        except Exception as e:
            raise ValueError(f"Error in removing stop words: {e}")

    def text_pipeline(self) -> str:
        """
        Pipeline lengkap untuk memproses teks dari huruf kecil, 
        pembersihan tanda baca, hingga menghapus stopwords.

        Returns
        -------
        str
            Teks yang telah diproses secara menyeluruh.
        """
        text1 = self.lower_case(self.text)
        text2 = self.cleaning_text(text1)
        text3 = self.remove_stop_words(text2, self.lang)
        return text3


def pipeline(input_text: str, lang: str='indonesian') -> str:
    """
    Fungsi pembungkus untuk memproses teks menggunakan class Preprocessing.

    Parameters
    ----------
    input_text : str
        Teks deskripsi yang ingin diproses.
    lang : str, optional
        Bahasa stopwords yang digunakan (default 'indonesian').

    Returns
    -------
    str
        Teks hasil praproses (lowercase, bersih tanda baca, tanpa stopwords).
    """
    try:
        clean_text = Preprocessing(input_text=input_text, language=lang)
        result = clean_text.text_pipeline()
        return result
    except Exception as e:
        raise RuntimeError(f"Error in text preprocessing: {e}")
