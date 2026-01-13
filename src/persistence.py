import os
import joblib

def guardar_modelo(modelo, vectorizer, nombre_modelo='sentiment_model.joblib', nombre_vectorizer='vectorizer.joblib'):
    """
    Guarda el modelo entrenado y el vectorizador (TF-IDF/BoW) en la carpeta models.
    """
    # Definir rutas absolutas/relativas a la carpeta models
    ruta_modelo = os.path.join('models', nombre_modelo)
    ruta_vec = os.path.join('models', nombre_vectorizer)

    # Crear directorio si no existe
    os.makedirs('models', exist_ok=True)

    try:
        joblib.dump(modelo, ruta_modelo)
        joblib.dump(vectorizer, ruta_vec)
        print(f"[SUCCESS] Modelo y vectorizador exportados a la carpeta 'models/'")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el modelo: {e}")

def cargar_modelo(nombre_modelo='sentiment_model.joblib', nombre_vectorizer='vectorizer.joblib'):
    """
    Carga los artefactos necesarios para la inferencia.
    """
    ruta_modelo = os.path.join('models', nombre_modelo)
    ruta_vec = os.path.join('models', nombre_vectorizer)

    if not os.path.exists(ruta_modelo) or not os.path.exists(ruta_vec):
        raise FileNotFoundError(f"No se encontraron los archivos en {ruta_modelo} o {ruta_vec}. Ejecuta train.py primero.")

    try:
        modelo = joblib.load(ruta_modelo)
        vectorizer = joblib.load(ruta_vec)
        print(f"[INFO] Artefactos cargados correctamente.")
        return modelo, vectorizer
    except Exception as e:
        print(f"[ERROR] Error al cargar los archivos .joblib: {e}")
        raise