def cargar_palabras(ruta):
    """
    Lee un archivo y devuelve una lista con las palabras, eliminando saltos de línea.
    """
    with open(ruta, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

def cargar_mensajes(ruta):
    """
    Lee un archivo de mensajes y devuelve una lista con los mensajes.
    """
    with open(ruta, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]
