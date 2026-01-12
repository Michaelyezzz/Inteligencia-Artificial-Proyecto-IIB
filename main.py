# main.py
from src.model_nb import predecir_sentimiento
from src.persistence import cargar_modelo
from src.io_data import cargar_palabras, cargar_mensajes
from src.preprocessing import limpiar_texto
from src.features import obtener_bow
from src.model_nb import crear_etiquetas, entrenar_modelo
from src.persistence import guardar_modelo

# Cargar los archivos de palabras y mensajes
positivas = cargar_palabras('data/raw/positivas.txt')
negativas = cargar_palabras('data/raw/negativas.txt')
mensajes = cargar_mensajes('data/raw/mensajes_prueba.txt')

# Limpiar los mensajes
mensajes_limpios = [limpiar_texto(mensaje) for mensaje in mensajes]

# Crear las etiquetas para los mensajes (positivo=1, negativo=0)
etiquetas = crear_etiquetas(mensajes_limpios, positivas, negativas)

# Obtener la representación BoW de los mensajes limpios
X, vectorizer = obtener_bow(mensajes_limpios)

# Entrenar el modelo Naive Bayes con alpha ajustado
modelo = entrenar_modelo(X, etiquetas, alpha=0.5)

# Guardar el modelo y el vectorizador
guardar_modelo(modelo, vectorizer)

# Cargar el modelo y el vectorizador guardados
modelo_cargado, vectorizer_cargado = cargar_modelo()

# Hacer una predicción sobre un mensaje de ejemplo
mensaje = "Me siento increíblemente feliz con el resultado"
predecir_sentimiento(mensaje, modelo_cargado, vectorizer_cargado)
