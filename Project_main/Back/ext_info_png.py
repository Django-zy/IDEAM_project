import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Cargar la imagen PNG
imagen_path = "D:/programacion/proyecto TPI/IDEAM_project/imagenes_png/20250223225710-6057ed9f246d3f6ce6ee07fd54182bd06eeef86e.png"  # Cambia esto por la ruta de tu imagen
imagen = Image.open(imagen_path)

# Convertir la imagen a una matriz numérica (RGB)
imagen_rgb = np.array(imagen)

# Mostrar la imagen para verificar
plt.imshow(imagen_rgb)
plt.title("Imagen de PM2.5")
plt.axis('off')
plt.show()

# Supongamos que tienes una escala de colores y sus valores correspondientes
# Por ejemplo, en tu imagen, los colores representan valores de PM2.5
# Debes crear un diccionario que mapee colores a valores de PM2.5
# Esto depende de la escala de colores de tu imagen.

# Ejemplo de mapeo de colores (ajusta esto según tu imagen)
escala_colores = {
    (0, 0, 255): 5,    # Azul: 5 µg/m³
    (0, 255, 0): 50,   # Verde: 50 µg/m³
    (255, 255, 0): 100, # Amarillo: 100 µg/m³
    (255, 0, 0): 200,  # Rojo: 200 µg/m³
}

# Función para encontrar el valor de PM2.5 más cercano a un color dado
def encontrar_valor_pm25(color, escala_colores):
    # Calcular la distancia entre el color y los colores de la escala
    distancias = [np.linalg.norm(np.array(color) - np.array(c)) for c in escala_colores.keys()]
    # Obtener el color más cercano
    color_mas_cercano = list(escala_colores.keys())[np.argmin(distancias)]
    # Devolver el valor de PM2.5 correspondiente
    return escala_colores[color_mas_cercano]

# Convertir la imagen RGB a una matriz de valores de PM2.5
pm25_matrix = np.zeros((imagen_rgb.shape[0], imagen_rgb.shape[1]))

for i in range(imagen_rgb.shape[0]):
    for j in range(imagen_rgb.shape[1]):
        color = tuple(imagen_rgb[i, j, :3])  # Ignorar el canal alfa si existe
        pm25_matrix[i, j] = encontrar_valor_pm25(color, escala_colores)

# Calcular el promedio de PM2.5
pm25_promedio = np.nanmean(pm25_matrix)
print(f"El promedio de PM2.5 en la imagen es: {pm25_promedio} µg/m³")

# Mostrar la matriz de PM2.5 como una imagen
plt.imshow(pm25_matrix, cmap='viridis')
plt.colorbar(label='Concentración de PM2.5 (µg/m³)')
plt.title("Matriz de PM2.5")
plt.show()