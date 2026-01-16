# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from src.model_nb import entrenar_modelo
from src.features import obtener_features
from src.preprocessing import limpiar_texto
from src.persistence import guardar_modelo
from src.evaluation import evaluar_modelo

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
    X_all, vectorizer = obtener_features(mensajes_limpios, debug=True, debug_samples=3, debug_idx=0, top_k=10)
    etiquetas = df['etiqueta'].tolist()

    # 2.1 Split para evaluación (hold-out)
    # Separamos antes de vectorizar para evitar fuga de información (TF-IDF debe ajustarse solo con entrenamiento)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        mensajes_limpios,
        etiquetas,
        test_size=0.2,
        random_state=42,
        stratify=etiquetas if len(set(etiquetas)) > 1 else None,
    )

    # 3. Optimización de Hiperparámetros (Búsqueda del mejor Alpha)
    print("Buscando el mejor parámetro Alpha...")
    X_train, vectorizer_eval = obtener_features(X_train_text)
    
    mejor_acc = 0
    mejor_alpha = 0.1
    
    # Probamos diferentes niveles de suavizado para encontrar el equilibrio
    for a in [0.01, 0.05, 0.1, 0.5, 1.0]:
        scores = cross_val_score(MultinomialNB(alpha=a), X_train, y_train, cv=5)
        mean_score = scores.mean()
        if mean_score > mejor_acc:
            mejor_acc = mean_score
            mejor_alpha = a
            
    print(f"Resultado: Alpha={mejor_alpha} seleccionado con {mejor_acc:.2%} de precisión media.")

    # 4. Entrenamiento + Evaluación (hold-out)
    print("\n--- Evaluación del modelo (hold-out 20%) ---")
    modelo_eval = entrenar_modelo(X_train, y_train, alpha=mejor_alpha)
    X_test = vectorizer_eval.transform(X_test_text)
    evaluar_modelo(modelo_eval, X_test, y_test)

    # 5. Re-entrenamiento final con TODO el dataset (para exportar el mejor modelo posible)
    print("\n--- Re-entrenando con todo el dataset para exportación ---")
    X_all, vectorizer = obtener_features(mensajes_limpios)
    modelo_final = entrenar_modelo(X_all, etiquetas, alpha=mejor_alpha)

    # 6. Validación de Sesgo
    analizar_sesgo_intensificadores(modelo_final, vectorizer)

    # 7. Guardar el modelo ganador
    guardar_modelo(modelo_final, vectorizer)
    print("\n--- Entrenamiento finalizado. Modelo optimizado y exportado ---")

if __name__ == "__main__":
    run_train()