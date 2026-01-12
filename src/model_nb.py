from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocessing import limpiar_texto

def crear_etiquetas(mensajes, positivas, negativas):
    """
    Crea etiquetas para los mensajes, basadas en la cantidad de palabras positivas y negativas.
    """
    etiquetas = []
    
    for mensaje in mensajes:
        # Contamos las ocurrencias de palabras positivas y negativas en cada mensaje
        contador_positivo = sum(1 for palabra in mensaje.split() if palabra in positivas)
        contador_negativo = sum(1 for palabra in mensaje.split() if palabra in negativas)
        
        # Si hay más palabras positivas que negativas, etiquetamos como 1 (positivo)
        if contador_positivo > contador_negativo:
            etiquetas.append(1)
        else:
            etiquetas.append(0)
    
    return etiquetas

def entrenar_modelo(X, y, alpha=1.0):
    """
    Entrena un modelo Naive Bayes para clasificación de sentimientos con el parámetro alpha ajustado.
    """
    # Dividir los datos en entrenamiento y prueba (80% para entrenar, 20% para probar)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar y entrenar el modelo Naive Bayes con el parámetro alpha ajustado
    modelo = MultinomialNB(alpha=alpha)
    modelo.fit(X_train, y_train)

    # Realizar predicciones sobre el conjunto de prueba
    predicciones = modelo.predict(X_test)

    # Evaluar el modelo
    precision = accuracy_score(y_test, predicciones)
    print(f"Precisión del modelo: {precision * 100:.2f}%")

    return modelo

def predecir_sentimiento(mensaje, modelo, vectorizer):
    """
    Función para predecir el sentimiento de un mensaje dado el modelo y vectorizador entrenados.
    """
    mensaje_vectorizado = vectorizer.transform([mensaje])  # Convertir el mensaje a su representación BoW
    probabilidad = modelo.predict_proba(mensaje_vectorizado)

    # Mapear probabilidades según las clases presentes en el modelo
    probabilidad_positivo = 0.0
    probabilidad_negativo = 0.0
    for idx, cls in enumerate(modelo.classes_):
        if cls == 1:
            probabilidad_positivo = probabilidad[0][idx]
        elif cls == 0:
            probabilidad_negativo = probabilidad[0][idx]

    # Predicción basada en la clase con mayor probabilidad
    if probabilidad_positivo > probabilidad_negativo:
        prediccion = "Positivo"
    else:
        prediccion = "Negativo"

    print(f"Probabilidades devueltas por el modelo: Positivo = {probabilidad_positivo:.2f}, Negativo = {probabilidad_negativo:.2f}")
    print(f"Predicción: {prediccion}")





