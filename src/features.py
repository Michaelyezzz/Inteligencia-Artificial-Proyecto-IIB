# src/features.py
from sklearn.feature_extraction.text import CountVectorizer

def obtener_bow(mensajes):
    """
    Convierte una lista de mensajes en una matriz de características usando Bag of Words (BoW).
    """
    # Usamos n-gramas de tamaño 1 y 2 (unigramas y bigramas)
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=1000)
    X = vectorizer.fit_transform(mensajes)  # Convierte los mensajes a BoW

    return X, vectorizer
