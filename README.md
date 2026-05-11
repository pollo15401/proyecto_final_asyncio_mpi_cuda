# Proyecto Final: Asyncio + MPI + NVIDIA CUDA para análisis de tráfico aéreo y ambiente

## Descripción
Sistema en Python para analizar tráfico aéreo y condiciones ambientales usando procesamiento paralelo. Integra las APIs AirLabs, OpenWeather, OpenAQ y ThingSpeak. El sistema consulta datos concurrentemente con Asyncio, distribuye el procesamiento con MPI4Py, acelera cálculos numéricos con CUDA/CuPy/Numba y presenta resultados en una interfaz Streamlit.

## APIs integradas
- **AirLabs**: vuelos activos, aerolínea, ruta, velocidad, altitud y estado.
- **OpenWeather**: temperatura, humedad, presión, lluvia, visibilidad y viento.
- **OpenAQ**: calidad del aire, PM2.5, PM10, NO2 y O3.
- **ThingSpeak**: lecturas IoT de sensores ambientales.

Si no tienes llaves API, el proyecto se ejecuta en modo demo con datos sintéticos controlados. Esto permite mostrar la arquitectura completa en clase sin fallar por falta de conexión o llaves.

## Instalación
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edita `.env` y coloca tus llaves reales:
```env
AIRLABS_API_KEY=...
OPENWEATHER_API_KEY=...
OPENAQ_API_KEY=...
THINGSPEAK_CHANNEL_ID=...
```

## Ejecución por consola
```bash
python main.py --city Chihuahua --lat 28.6353 --lon -106.0889 --airport CUU --hours 24
```

## Ejecución con MPI
```bash
mpiexec -n 4 python main.py --city Chihuahua --lat 28.6353 --lon -106.0889 --airport CUU --hours 24
```

## Interfaz tipo proyecto final
```bash
streamlit run app/dashboard.py
```

## Salidas generadas
- `salidas/csv/airlabs_vuelos.csv`
- `salidas/csv/openweather_clima.csv`
- `salidas/csv/openaq_calidad_aire.csv`
- `salidas/csv/thingspeak_sensores.csv`
- `salidas/csv/tabla_modelo_integrada.csv`
- `salidas/csv/outliers_detectados.csv`
- `salidas/graficas/*.png`
- `salidas/resumen_resultados.json`
- `salidas/resumen_ejecucion.txt`

## Subida a GitHub
```bash
git init
git add .
git commit -m "Estructura profesional del proyecto final"
git commit -m "Integra APIs con Asyncio y modo demo"
git commit -m "Agrega MPI, CUDA y dashboard Streamlit"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/proyecto-final-asyncio-mpi-cuda.git
git push -u origin main
```
