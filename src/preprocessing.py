# src/preprocessing.py
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Asegurarnos de que los recursos necesarios de NLTK estén disponibles.
def _ensure_nltk_resource(name, resource_path):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        try:
            nltk.download(name)
        except Exception:
            pass

# Recursos requeridos por tokenización y lematización
_ensure_nltk_resource('punkt', 'tokenizers/punkt')
_ensure_nltk_resource('wordnet', 'corpora/wordnet')
_ensure_nltk_resource('omw-1.4', 'corpora/omw-1.4')

# Inicializamos el lematizador
lemmatizer = WordNetLemmatizer()

def limpiar_texto(texto):
    """
    Limpiar el texto eliminando stopwords, caracteres especiales y lematizando.
    """
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Eliminar caracteres especiales y números
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    
    # Tokenizar el texto (separar en palabras)
    try:
        palabras = word_tokenize(texto)
    except LookupError:
        # Fallback simple si faltan recursos de NLTK: extraer palabras con regex
        palabras = re.findall(r'\b[a-zA-Z]+\b', texto)
    
    # Eliminar stopwords y lematizar
    palabras_limpias = [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra not in ENGLISH_STOP_WORDS]
    
    # Unir las palabras de nuevo en un único string
    texto_limpio = " ".join(palabras_limpias)
    
    return texto_limpio
