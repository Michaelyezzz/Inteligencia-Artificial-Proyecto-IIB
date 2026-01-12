import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer

def guardar_modelo(modelo, vectorizer, nombre_modelo='sentiment_model.joblib', nombre_vectorizer='vectorizer.joblib'):
    """
    Guarda el modelo y el vectorizador usando joblib.
    """
    # Asegurar que los archivos se guarden en la carpeta models por defecto
    if nombre_modelo == 'sentiment_model.joblib':
        nombre_modelo = os.path.join('models', 'sentiment_model.joblib')
    if nombre_vectorizer == 'vectorizer.joblib':
        nombre_vectorizer = os.path.join('models', 'vectorizer.joblib')

    # Crear directorio si no existe
    dir_model = os.path.dirname(nombre_modelo)
    if dir_model:
        os.makedirs(dir_model, exist_ok=True)

    joblib.dump(modelo, nombre_modelo)
    joblib.dump(vectorizer, nombre_vectorizer)
    print(f"Modelo y vectorizador guardados como {nombre_modelo} y {nombre_vectorizer}")

def cargar_modelo(nombre_modelo='sentiment_model.joblib', nombre_vectorizer='vectorizer.joblib'):
    """
    Carga el modelo y el vectorizador desde los archivos guardados.
    """
    # Intentar cargar desde models/ por defecto
    if nombre_modelo == 'sentiment_model.joblib':
        candidate_model = os.path.join('models', 'sentiment_model.joblib')
        if os.path.exists(candidate_model):
            nombre_modelo = candidate_model
    if nombre_vectorizer == 'vectorizer.joblib':
        candidate_vec = os.path.join('models', 'vectorizer.joblib')
        if os.path.exists(candidate_vec):
            nombre_vectorizer = candidate_vec

    modelo = joblib.load(nombre_modelo)
    vectorizer = joblib.load(nombre_vectorizer)
    print(f"Modelo y vectorizador cargados desde {nombre_modelo} y {nombre_vectorizer}")
    return modelo, vectorizer

def guardar_boW_features(features, filename='data/processed/bow_features.pkl'):
    """
    Guarda las características BoW procesadas en un archivo.
    """
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    joblib.dump(features, filename)
    print(f"Características BoW guardadas en {filename}")
    
def obtener_bow(mensajes):
    """
    Convierte una lista de mensajes en una matriz de características usando Bag of Words (BoW).
    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(mensajes)  # Convierte los mensajes a BoW

    # Guardar las características BoW procesadas
    guardar_boW_features(X)

    return X, vectorizer
