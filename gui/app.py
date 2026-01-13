# gui/app.py
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys

# Ajuste de rutas
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.persistence import cargar_modelo
from src.preprocessing import limpiar_texto
from gui.widgets import configurar_estilos # Usando tus widgets

class AppSentimientos:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("SentimAI - Analizador Profesional")
        self.ventana.geometry("500x600")
        self.ventana.configure(bg="#2c3e50") # Fondo azul oscuro profesional
        
        # Carga de modelo
        self.modelo, self.vectorizer = cargar_modelo()
        self.clases = self.modelo.classes_

        # Encabezado
        self.header = tk.Label(ventana, text="Análisis de Sentimiento IA", 
                              fg="#ecf0f1", bg="#2c3e50", font=("Segoe UI", 20, "bold"))
        self.header.pack(pady=30)

        # Entrada de texto redondeada (simulada con relief)
        self.entrada = tk.Entry(ventana, width=35, font=("Segoe UI", 12), 
                               relief="flat", bg="#ecf0f1", fg="#2c3e50")
        self.entrada.pack(pady=10, ipady=8)
        self.entrada.insert(0, "Escribe tu mensaje aquí...")
        self.entrada.bind("<FocusIn>", lambda e: self.entrada.delete(0, tk.END))

        # Botón con estilo
        self.btn_analizar = tk.Button(ventana, text="ANALIZAR AHORA", command=self.analizar,
                                     bg="#3498db", fg="white", font=("Segoe UI", 11, "bold"),
                                     activebackground="#2980b9", relief="flat", cursor="hand2")
        self.btn_analizar.pack(pady=25, ipadx=20, ipady=5)

        # Área de resultados (Tarjeta)
        self.card = tk.Frame(ventana, bg="#34495e", padx=20, pady=20)
        self.card.pack(fill="x", padx=50)

        self.lbl_emoji = tk.Label(self.card, text="🤖", bg="#34495e", font=("Segoe UI", 40))
        self.lbl_emoji.pack()

        self.lbl_res = tk.Label(self.card, text="Esperando entrada", fg="#bdc3c7", 
                               bg="#34495e", font=("Segoe UI", 14, "bold"))
        self.lbl_res.pack(pady=10)

        # Barra de progreso para la confianza
        self.progress = ttk.Progressbar(self.card, length=200, mode='determinate')
        self.progress.pack(pady=10)

    def analizar(self):
        mensaje = self.entrada.get()
        if not mensaje or mensaje == "Escribe tu mensaje aquí...": return

        # Proceso técnico
        limpio = limpiar_texto(mensaje)
        vec = self.vectorizer.transform([limpio])
        probs = self.modelo.predict_proba(vec)[0]
        
        prob_map = dict(zip(self.clases, probs))
        p_pos, p_neg = prob_map.get(1, 0), prob_map.get(0, 0)

        # UI Update dinámico
        es_positivo = p_pos > p_neg
        texto = "POSITIVO" if es_positivo else "NEGATIVO"
        color = "#2ecc71" if es_positivo else "#e74c3c"
        emoji = "😊" if es_positivo else "☹️"
        confianza = max(p_pos, p_neg)

        self.lbl_res.config(text=f"{texto} ({confianza:.1%})", fg=color)
        self.lbl_emoji.config(text=emoji)
        self.progress['value'] = confianza * 100
        
        # Cambiar color del botón según resultado
        self.btn_analizar.config(bg=color)

if __name__ == "__main__":
    root = tk.Tk()
    app = AppSentimientos(root)
    root.mainloop()