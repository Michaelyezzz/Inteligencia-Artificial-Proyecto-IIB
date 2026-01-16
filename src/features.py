from sklearn.feature_extraction.text import TfidfVectorizer

# Puedes definir una lista manual o usar una librería
stop_words_es = ['me', 'siento', 'muy', 'un', 'el', 'la', 'de','demasiado'] 

def obtener_features(mensajes, debug=False, debug_samples=3, debug_idx=0, top_k=10):

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),        # Solo unigramas y bigramas (más estable)
        stop_words=stop_words_es, # Ignora el ruido de palabras frecuentes
        max_features=800,          # Reducimos para concentrar la 'inteligencia'
        sublinear_tf=True, 
        strip_accents='unicode',
        min_df=2,                  # DESCARTA palabras que solo salen 1 vez (limpia el ruido)
        max_df=0.8                 # Descarta palabras demasiado comunes
    )
    X = vectorizer.fit_transform(mensajes)

    if debug:
        print("\n--- DEBUG features.obtener_features ---")
        n = min(debug_samples, len(mensajes))
        print(f"Mensajes recibidos: {len(mensajes)} (mostrando {n})")
        for i in range(n):
            print(f"[{i}] {mensajes[i]}")

        print("\nResumen TF-IDF:")
        print("Shape (n_mensajes, n_features):", X.shape)
        print("Valores no-cero (nnz):", X.nnz)
        feature_names = vectorizer.get_feature_names_out()
        print("Primeras features:", list(feature_names[:20]))

        if 0 <= debug_idx < X.shape[0]:
            row = X[debug_idx].tocoo()
            pares = [(feature_names[j], float(v)) for j, v in zip(row.col, row.data)]
            pares_ordenados = sorted(pares, key=lambda x: x[1], reverse=True)
            print(f"\nTop {top_k} términos del mensaje idx={debug_idx}:")
            for term, weight in pares_ordenados[:top_k]:
                print(f"  {term}: {weight:.4f}")
        else:
            print(f"\n[WARN] debug_idx={debug_idx} fuera de rango; usa 0..{X.shape[0]-1}.")

    return X, vectorizer