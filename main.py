# main.py
import os
import sys
import tkinter as tk
from gui.app import AppSentimientos

def launch():
    """
    Punto de entrada principal para el Clasificador de Sentimientos.
    Verifica la existencia del modelo antes de iniciar la interfaz.
    """
    print("--- Iniciando Clasificador de Sentimientos ---")
    
    # Verificación técnica de artefactos
    modelo_path = os.path.join('models', 'sentiment_model.joblib')
    vectorizador_path = os.path.join('models', 'vectorizer.joblib')
    
    if not os.path.exists(modelo_path) or not os.path.exists(vectorizador_path):
        print("[ERROR] No se detectó un modelo entrenado.")
        print("Por favor, ejecuta 'python train.py' para generar los archivos necesarios.")
        return

    # Iniciar la interfaz gráfica
    root = tk.Tk()
    app = AppSentimientos(root)
    # Centrar la ventana en la pantalla
    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    if w <= 1 and h <= 1:
        w = root.winfo_reqwidth()
        h = root.winfo_reqheight()
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")

    root.mainloop()

if __name__ == "__main__":
    launch()