import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Cargar la imagen
imagen_path = "D:/programacion/proyecto TPI/IDEAM_project/imagenes_png/20250223225710-6057ed9f246d3f6ce6ee07fd54182bd06eeef86e.png"
imagen = Image.open(imagen_path)

# Convertir la imagen a una matriz numérica (RGB)
imagen_rgb = np.array(imagen)

# Mostrar la imagen original
plt.imshow(imagen_rgb)
plt.title("Imagen Original de PM2.5")
plt.axis('off')
plt.show()

# Recortar la imagen para Bogotá (aproximadamente en el centro de Colombia)
# Esto es un estimado, ya que sin coordenadas exactas, se hace con base en observación.
y1, y2 = int(imagen_rgb.shape[0] * 0.40), int(imagen_rgb.shape[0] * 0.55)
x1, x2 = int(imagen_rgb.shape[1] * 0.30), int(imagen_rgb.shape[1] * 0.40)
imagen_bogota = imagen_rgb[y1:y2, x1:x2]

# Mostrar la imagen recortada para verificar que está enfocada en Bogotá
plt.imshow(imagen_bogota)
plt.title("Zona Recortada de Bogotá")
plt.axis('off')
plt.show()

# Extraer colores únicos en la imagen para analizar la escala
colores_unicos = np.unique(imagen_bogota.reshape(-1, imagen_bogota.shape[2]), axis=0)

# Mostrar los colores únicos detectados
colores_unicos

# Definir la escala de colores basada en la barra de referencia en la imagen
escala_colores_pm25 = {
    (181, 226, 179): 20,   # Verde claro
    (123, 201, 124): 30,   # Verde medio
    (65, 171, 93): 40,     # Verde oscuro
    (35, 139, 69): 50,     # Verde más oscuro
    (116, 196, 118): 60,   # Verde azulado
    (49, 163, 84): 80,     # Verde más intenso
    (161, 217, 155): 100,  # Amarillo-verde
    (254, 217, 118): 150,  # Amarillo
    (252, 141, 89): 200,   # Naranja
    (227, 74, 51): 300,    # Rojo
    (179, 0, 0): 500       # Rojo oscuro
}

# Función para encontrar el valor de PM2.5 más cercano a un color dado
def encontrar_valor_pm25(color, escala_colores_pm25):
    # Calcular la distancia entre el color y los colores de la escala
    distancias = [np.linalg.norm(np.array(color) - np.array(c)) for c in escala_colores_pm25.keys()]
    # Obtener el color más cercano
    color_mas_cercano = list(escala_colores_pm25.keys())[np.argmin(distancias)]
    # Devolver el valor de PM2.5 correspondiente
    return escala_colores_pm25[color_mas_cercano]

# Convertir la imagen recortada a una matriz de valores de PM2.5
pm25_matrix_bogota = np.zeros((imagen_bogota.shape[0], imagen_bogota.shape[1]))

for i in range(imagen_bogota.shape[0]):
    for j in range(imagen_bogota.shape[1]):
        color = tuple(imagen_bogota[i, j, :3])  # Ignorar el canal alfa si existe
        pm25_matrix_bogota[i, j] = encontrar_valor_pm25(color, escala_colores_pm25)

# Calcular el promedio de PM2.5 en la zona de Bogotá
pm25_promedio_bogota = np.nanmean(pm25_matrix_bogota)
print(f"{pm25_promedio_bogota} µg/m³")