from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocessing import limpiar_texto
from sklearn.model_selection import cross_val_score, train_test_split

def crear_etiquetas(mensajes, positivas, negativas):
    etiquetas = []
    for i, mensaje in enumerate(mensajes):
        palabras = mensaje.split()
        # Buscamos coincidencias exactas con los diccionarios limpios
        cp = sum(1 for p in palabras if p in positivas)
        cn = sum(1 for p in palabras if p in negativas)
        
        # Aumentamos la sensibilidad: si hay empate, el modelo es neutro.
        # Para el entrenamiento, es mejor ser estricto.
        if cp > cn:
            etiquetas.append(1)
        elif cn > cp:
            etiquetas.append(0)
        else:
            # Si no hay claridad, usamos un balanceo estadístico (i % 2)
            etiquetas.append(i % 2)
    return etiquetas

def entrenar_modelo(X, y, alpha= 1.0):
    # Evaluación robusta con K-Fold Cross Validation (K=5)
    scores = cross_val_score(MultinomialNB(alpha=alpha), X, y, cv=5)
    print(f"Precisión media (Cross-Validation): {scores.mean() * 100:.2f}% (+/- {scores.std() * 2:.2f}%)")
    
    # Entrenamiento final con todos los datos disponibles
    modelo = MultinomialNB(alpha=alpha)
    modelo.fit(X, y)
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





