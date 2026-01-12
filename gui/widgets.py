import tkinter as tk

def crear_label(ventana, texto, fila, columna):
    """
    Función para crear una etiqueta en la interfaz gráfica.
    """
    label = tk.Label(ventana, text=texto, font=('Arial', 12))
    label.grid(row=fila, column=columna)
    return label

def crear_boton(ventana, texto, comando, fila, columna):
    """
    Función para crear un botón en la interfaz gráfica.
    """
    boton = tk.Button(ventana, text=texto, command=comando, font=('Arial', 12))
    boton.grid(row=fila, column=columna)
    return boton
