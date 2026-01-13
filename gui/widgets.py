# gui/widgets.py
import tkinter as tk
from tkinter import ttk

def configurar_estilos():
    """Configura temas y fuentes globales."""
    style = ttk.Style()
    style.theme_use('clam') # Un tema base más limpio
    style.configure("TButton", font=("Segoe UI", 10), padding=10)
    style.configure("TLabel", font=("Segoe UI", 11), background="#2c3e50", foreground="white")

def crear_tarjeta_resultado(parent):
    """Crea un contenedor estilizado para mostrar los resultados."""
    frame = tk.Frame(parent, bg="#34495e", bd=0, highlightthickness=1, highlightbackground="#5d6d7e")
    return frame