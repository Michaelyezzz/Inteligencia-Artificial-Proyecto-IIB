# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from src.model_nb import entrenar_modelo
from src.features import obtener_features
from src.preprocessing import limpiar_texto
from src.persistence import guardar_modelo

def analizar_sesgo_intensificadores(modelo, vectorizer):
    """
    Detecta si palabras como 'muy' distorsionan la predicción.
    """
    print("\n--- Análisis de Sesgo (Sanity Check) ---")
    pruebas = [
        ("deprimido", "muy deprimido"),
        ("triste", "muy triste"),
        ("feliz", "muy feliz"),
        ("bueno", "muy bueno")
    ]
    
    for base, intensificado in pruebas:
        X_base = vectorizer.transform([limpiar_texto(base)])
        X_int  = vectorizer.transform([limpiar_texto(intensificado)])
        
        # Obtenemos probabilidad de la clase Positiva (índice 1)
        prob_base = modelo.predict_proba(X_base)[0][1]
        prob_int  = modelo.predict_proba(X_int)[0][1]
        
        status = "[OK]" if (prob_base < 0.5 and prob_int < 0.5) or (prob_base > 0.5 and prob_int > 0.5) else "[ALERTA DE SESGO]"
        print(f"'{base}' ({prob_base:.2%}) -> '{intensificado}' ({prob_int:.2%}) {status}")

def run_train():
    print("--- Iniciando Entrenamiento Supervisado Avanzado ---")
    
    # 1. Cargar el dataset etiquetado
    try:
        df = pd.read_csv('data/raw/dataset_final.csv', sep='|')
    except Exception as e:
        print(f"[ERROR] No se pudo cargar dataset_final.csv: {e}")
        return

    # 2. Preprocesamiento
    print("Preprocesando mensajes...")
    mensajes_limpios = [limpiar_texto(str(m)) for m in df['mensaje']]
    etiquetas = df['etiqueta'].tolist()

    # 3. Optimización de Hiperparámetros (Búsqueda del mejor Alpha)
    print("Buscando el mejor parámetro Alpha...")
    X, vectorizer = obtener_features(mensajes_limpios) #
    
    mejor_acc = 0
    mejor_alpha = 0.1
    
    # Probamos diferentes niveles de suavizado para encontrar el equilibrio
    for a in [0.01, 0.05, 0.1, 0.5, 1.0]:
        scores = cross_val_score(MultinomialNB(alpha=a), X, etiquetas, cv=5)
        mean_score = scores.mean()
        if mean_score > mejor_acc:
            mejor_acc = mean_score
            mejor_alpha = a
            
    print(f"Resultado: Alpha={mejor_alpha} seleccionado con {mejor_acc:.2%} de precisión media.")

    # 4. Entrenamiento final con el mejor Alpha
    modelo_final = entrenar_modelo(X, etiquetas, alpha=mejor_alpha)

    # 5. Validación de Sesgo
    analizar_sesgo_intensificadores(modelo_final, vectorizer)

    # 6. Guardar el modelo ganador
    guardar_modelo(modelo_final, vectorizer)
    print("\n--- Entrenamiento finalizado. Modelo optimizado y exportado ---")

if __name__ == "__main__":
    run_train()