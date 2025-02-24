import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ruta al archivo NetCDF
nc_file = "D:/programacion/proyecto TPI/IDEAM_project/imagenes/PM25_Colombia.nc"

# Cargar el archivo NetCDF
ds = xr.open_dataset(nc_file)

# # Mostrar las variables disponibles en el dataset
# print(ds)

# Convertir el dataset a un DataFrame de Pandas
df = ds.to_dataframe().reset_index()

# # Mostrar las primeras filas del DataFrame
# print(df.head())

# Filtrar datos no nulos
df_filtered = df.dropna(subset=['PM2.5'])

# Crear una tabla pivote para el mapa de calor
heatmap_data = df_filtered.pivot_table(index='lat', columns='lon', values='PM2.5')

pm25_promedio = ds['PM2.5'].mean().values
print(f"El promedio de PM2.5 en Bogotá es: {pm25_promedio} µg/m³")

# print("Promedio excluyente")
# pm25_promedio = ds['PM2.5'].where(ds['PM2.5'] >= 0).mean().values
# print(pm25_promedio)

# Crear el mapa de calor
plt.figure(figsize=(10, 8))
plt.imshow(heatmap_data, origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Concentración de PM2.5')
plt.title('Mapa de concentracion de PM2.5 en Colombia')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.show()
