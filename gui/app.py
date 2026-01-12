# src/gui/app.py
import sys
import os

# Calculamos la ruta absoluta de la carpeta 'src/' desde la carpeta 'gui/'
# Obtenemos la ruta de la carpeta raíz del proyecto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Esto da la ruta a "Inteligencia Artificial Proyecto IIB"
src_path = os.path.join(project_root, 'src')  # Ruta a la carpeta 'src'

# Añadir la raíz del proyecto a `sys.path` para que `import src...` funcione.
# Si añadimos la carpeta `src` directamente, Python buscará un paquete `src` dentro
# de esa carpeta (no dentro de su padre), lo que provoca el error.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Verificamos que la carpeta 'src' se haya añadido a sys.path
print(f"Rutas en sys.path: {sys.path}")

# Ahora podemos importar desde 'src/'
import tkinter as tk
from tkinter import messagebox
from src.model_nb import predecir_sentimiento  # Importar desde src
from src.persistence import cargar_modelo

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Clasificador de Sentimientos")

# Cargar el modelo y el vectorizador
modelo, vectorizer = cargar_modelo()

# Función para clasificar el mensaje ingresado
def clasificar_mensaje():
    mensaje = entrada_texto.get()  # Obtener el texto ingresado
    if not mensaje.strip():  # Verificar si el texto está vacío
        messagebox.showerror("Error", "Por favor, ingresa un mensaje.")
        return
    
    # Llamar a la función de predicción
    predecir_sentimiento(mensaje, modelo, vectorizer)

# Etiqueta para instrucciones
etiqueta_instrucciones = tk.Label(ventana, text="Escribe un mensaje y presiona 'Clasificar'.")
etiqueta_instrucciones.pack(pady=10)

# Caja de texto para ingresar el mensaje
entrada_texto = tk.Entry(ventana, width=50)
entrada_texto.pack(pady=10)

# Botón para clasificar el mensaje
boton_clasificar = tk.Button(ventana, text="Clasificar", command=clasificar_mensaje)
boton_clasificar.pack(pady=20)

# Ejecutar la interfaz gráfica
ventana.mainloop()
