# Proyecto Final: Análisis de Tráfico Aéreo con Asyncio + MPI + CUDA

Sistema profesional en Python para análisis de tráfico aéreo y condiciones ambientales usando procesamiento paralelo y distribuido.

## 🎯 Características

- **Asyncio**: Consulta concurrente de 4 APIs (AirLabs, OpenWeather, OpenAQ, ThingSpeak)
- **MPI4Py**: Distribución del procesamiento de vuelos entre múltiples procesos
- **CUDA/CuPy/Numba**: Aceleración GPU para correlaciones y detección de outliers
- **Pandas**: Integración y análisis tabular de datos
- **Streamlit + Plotly**: Dashboard interactivo con visualizaciones en tiempo real
- **Matplotlib**: Gráficas estáticas exportables

## 📊 Fuentes de Datos

| API | Datos Obtenidos |
|-----|----------------|
| **AirLabs** | Vuelos, horarios, retrasos, cancelaciones, velocidad, altitud |
| **OpenWeather** | Temperatura, humedad, presión, lluvia, visibilidad, viento |
| **OpenAQ** | Calidad del aire (PM2.5, PM10, NO2, O3) |
| **ThingSpeak** | Sensores IoT (8 campos: temp, humedad, presión, PM2.5, PM10, luz, viento) |

**Modo Demo**: Si no hay API keys, el sistema genera datos sintéticos realistas para demostración.

## 🚀 Instalación

```bash
# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Configurar API keys
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac
```

Edita `.env` con tus llaves reales:
```env
AIRLABS_API_KEY=tu_key_aqui
OPENWEATHER_API_KEY=tu_key_aqui
OPENAQ_API_KEY=tu_key_aqui  # Opcional
THINGSPEAK_CHANNEL_ID=12397
THINGSPEAK_READ_API_KEY=    # Opcional para canal público
DEFAULT_CITY=Chihuahua
DEFAULT_LAT=28.6353
DEFAULT_LON=-106.0889
DEFAULT_AIRPORT=CUU
```

## 💻 Uso

### Ejecución por consola (modo simple)
```bash
python main.py --city Chihuahua --lat 28.6353 --lon -106.0889 --airport CUU --hours 24
```

### Ejecución con MPI (procesamiento distribuido)
```bash
mpiexec -n 4 python main.py --city Chihuahua --lat 28.6353 --lon -106.0889 --airport CUU --hours 24
```

### Dashboard interactivo (recomendado)
```bash
streamlit run app/dashboard.py
```
Abre http://localhost:8504 en tu navegador.

## 📁 Estructura del Proyecto

```
proyecto_final_asyncio_mpi_cuda/
├── app/
│   └── dashboard.py          # Dashboard Streamlit interactivo
├── src/
│   ├── api_clients.py        # Clientes asíncronos para APIs
│   ├── config.py             # Configuración y settings
│   ├── gpu_compute.py        # Cálculos acelerados con CUDA/CuPy
│   ├── mpi_processing.py     # Distribución con MPI4Py
│   ├── pipeline.py           # Orquestador principal
│   └── visualization.py      # Gráficas Matplotlib
├── salidas/
│   ├── csv/                  # Datos exportados
│   ├── graficas/             # Gráficas PNG
│   ├── resumen_resultados.json
│   └── resumen_ejecucion.txt
├── main.py                   # Entrada por consola
├── requirements.txt
├── .env
└── README.md
```

## 📈 Salidas Generadas

### CSV
- `airlabs_vuelos.csv` — Vuelos con retrasos, cancelaciones, velocidad, altitud
- `openweather_clima.csv` — Datos meteorológicos por hora
- `openaq_calidad_aire.csv` — Calidad del aire (PM2.5, PM10, NO2, O3)
- `thingspeak_sensores.csv` — Lecturas de sensores IoT
- `tabla_modelo_integrada.csv` — Tabla unificada clima + vuelos + calidad aire
- `outliers_detectados.csv` — Vuelos atípicos detectados con CUDA

### Gráficas (PNG)
- `vuelos_por_aerolinea.png`
- `retraso_por_hora.png`
- `lluvia_vs_retraso.png`
- `rutas_frecuentes.png`

### Reportes
- `resumen_resultados.json` — Métricas completas en JSON
- `resumen_ejecucion.txt` — Resumen legible

## 🔬 Análisis Implementados

### Preguntas de Investigación
1. **¿Hay relación entre clima y retrasos?**
   - Correlaciones calculadas con CUDA/CuPy
   - Scatter plots con líneas de tendencia

2. **¿Aumentan los retrasos con lluvia o baja visibilidad?**
   - Análisis de correlación lluvia → retraso
   - Análisis visibilidad → retraso

3. **¿Qué aeropuertos presentan mayor cantidad de retrasos?**
   - Ranking de retrasos promedio por aeropuerto de origen
   - Gráficas de barras comparativas

4. **¿Cómo cambia la actividad aérea durante el día?**
   - Vuelos activos por hora del día
   - Retraso promedio por hora del día

5. **¿Qué condiciones ambientales coinciden con mayor tráfico?**
   - Scatter plots variables ambientales vs vuelos activos
   - Análisis de tendencias

### Métricas Calculadas
- Total de vuelos procesados
- Vuelos cancelados
- Retraso promedio global y por aeropuerto
- Velocidad y altitud promedio
- Correlaciones clima → retraso (7 variables)
- Detección de outliers (z-score > 2.5)
- Rutas más frecuentes
- Vuelos por aerolínea

## 🛠️ Tecnologías

| Componente | Tecnología | Uso |
|------------|-----------|-----|
| **Concurrencia** | Asyncio + aiohttp | Consulta paralela de 4 APIs |
| **Distribución** | MPI4Py | Procesamiento distribuido de vuelos |
| **Aceleración GPU** | CUDA / CuPy / Numba | Correlaciones y detección de outliers |
| **Análisis** | Pandas + NumPy | Limpieza, integración y análisis |
| **Visualización** | Streamlit + Plotly + Matplotlib | Dashboard interactivo y gráficas |
| **Configuración** | python-dotenv | Gestión de API keys |

## 📊 Dashboard Interactivo

El dashboard Streamlit incluye 5 tabs:

1. **📊 Resumen**: Vuelos por aerolínea, rutas frecuentes, actividad por hora, retrasos por aeropuerto
2. **🛫 Vuelos**: Mapa interactivo, consulta de vuelo específico, simulación de rutas
3. **🌦️ Ambiente**: Clima vs retrasos, variables ambientales, sensores ThingSpeak, rosa de vientos
4. **⚡ CUDA/MPI**: Correlaciones, tiempos de ejecución, outliers detectados
5. **📁 Evidencias**: Stack tecnológico, comandos, archivos generados, preview de gráficas

## 🔧 Requisitos

- Python 3.10+
- CUDA Toolkit 12.x (opcional, para aceleración GPU)
- MPI implementation (opcional, para procesamiento distribuido)

## 📝 Notas

- **Datos reales**: El sistema usa `/schedules` de AirLabs para obtener retrasos y cancelaciones reales, luego enriquece con `/flights` para velocidad/altitud de vuelos activos.
- **Fallback automático**: Si una API falla, usa datos demo sin interrumpir la ejecución.
- **Velocidad corregida**: AirLabs devuelve velocidad en km/h directamente (no en nudos).
- **Correlaciones robustas**: Validación de varianza mínima para evitar divisiones por cero.
- **Compatible con Windows/Linux/Mac**.

## 🎓 Proyecto Final

Este proyecto demuestra:
- Programación asíncrona con Asyncio
- Procesamiento distribuido con MPI
- Aceleración GPU con CUDA/CuPy
- Integración de APIs REST
- Análisis de datos con Pandas
- Visualización interactiva con Streamlit
- Arquitectura modular y profesional

## 📄 Licencia

Proyecto académico — Cómputo Paralelo y Distribuido
