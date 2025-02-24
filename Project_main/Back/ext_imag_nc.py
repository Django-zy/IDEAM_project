import ee
import geemap
import os
import rasterio
import xarray as xr
import numpy as np

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/programacion/proyecto TPI/IDEAM_project/ideam-extraccion-e8cdba01ac1f.json"
# Autenticación en GEE
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='ideam-extraccion')

# Definir área de Colombia
# colombia = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
#     .filter(ee.Filter.eq('country_na', 'Colombia'))
bogota_coords = [
    [-74.21, 4.83],  # Esquina superior izquierda
    [-73.96, 4.83],  # Esquina superior derecha
    [-73.96, 4.48],  # Esquina inferior derecha
    [-74.21, 4.48],  # Esquina inferior izquierda
    [-74.21, 4.83]   # Volver al punto inicial para cerrar el polígono
]
bogota = ee.Geometry.Polygon(bogota_coords)

# Colección de datos de PM2.5 de CAMS
# cams_pm25 = ee.ImageCollection("ECMWF/CAMS/NRT") \
#     .select('pm2p5') \
#     .filterDate('2021-02-01', '2021-02-22')  # Modificar fecha según necesidad

cams_pm25 = ee.ImageCollection("ECMWF/CAMS/NRT") \
    .select('particulate_matter_d_less_than_25_um_surface') \
    .filterDate('2021-02-01', '2021-02-25')  # Ajusta las fechas según sea necesario

# Promedio diario
pm25_mean = cams_pm25.mean().clip(bogota)

# Exportar la imagen como GeoTIFF a Google Drive
task = ee.batch.Export.image.toDrive(
    image=pm25_mean,
    description='PM25_Colombia_Diario',
    folder='GEE_Exports',
    fileNamePrefix='PM25_Colombia',
    scale=10000,
    region=bogota.bounds(),#colombia.geometry().bounds(),
    fileFormat='GeoTIFF'
)
task.start()
print("Exportación iniciada a Google Drive...")

# ---- Conversión de GeoTIFF a NetCDF ----

def convert_geotiff_to_netcdf(tiff_file, nc_file):
    """ Convierte un archivo GeoTIFF a NetCDF """
    with rasterio.open(tiff_file) as src:
        data = src.read(1)  # Leer la primera banda
        transform = src.transform
        bounds = src.bounds
        res = src.res
        nodata = src.nodata

        # Definir coordenadas
        lon = np.arange(bounds.left + res[0] / 2, bounds.right, res[0])
        lat = np.arange(bounds.top - res[1] / 2, bounds.bottom, -res[1])

        # Crear Dataset de NetCDF con xarray
        ds = xr.Dataset(
            {
                "PM2.5": (["lat", "lon"], data)
            },
            coords={
                "lon": lon,
                "lat": lat
            },
        )

        # Guardar en NetCDF
        ds.to_netcdf(nc_file)
        print(f"Archivo NetCDF guardado en: {nc_file}")

# Ruta de los archivos (ajustar después de descargar desde Google Drive)
tiff_file = "D:/programacion/proyecto TPI/IDEAM_project/imagenes/PM25_Colombia (1).tif" #"PM25_Colombia.tif"  # Debe descargarse manualmente de Google Drive
nc_file = "D:/programacion/proyecto TPI/IDEAM_project/imagenes/PM25_Colombia.nc" # nc_file = "PM25_Colombia.nc"

# Convertir GeoTIFF a NetCDF
if os.path.exists(tiff_file):
    convert_geotiff_to_netcdf(tiff_file, nc_file)
else:
    print("Descarga primero el archivo GeoTIFF desde Google Drive antes de convertirlo.")