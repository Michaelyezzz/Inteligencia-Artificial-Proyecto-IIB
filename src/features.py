from sklearn.feature_extraction.text import TfidfVectorizer

# Puedes definir una lista manual o usar una librería
stop_words_es = ['me', 'siento', 'muy', 'un', 'el', 'la', 'de','demasiado'] 

def obtener_features(mensajes):
 
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
    return X, vectorizer