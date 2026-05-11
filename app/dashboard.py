import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import plotly.express as px
import streamlit as st

from src.config import Settings
from src.pipeline import run_pipeline

st.set_page_config(page_title="Proyecto Final | Asyncio + MPI + CUDA", layout="wide", page_icon="✈️")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
.big-card {border-radius: 18px; padding: 18px; background: #0f172a; color: white;}
.metric-card {border-radius: 16px; padding: 14px; background: #f8fafc; border: 1px solid #e2e8f0;}
</style>
""", unsafe_allow_html=True)

st.title("✈️ Dashboard de Tráfico Aéreo y Condiciones Ambientales")
st.caption("Proyecto final con Asyncio, MPI4Py, CUDA/CuPy/Numba, Pandas, APIs en la nube y visualización interactiva.")

with st.sidebar:
    st.header("Configuración")
    city = st.text_input("Ciudad", "Chihuahua")
    airport = st.text_input("Aeropuerto IATA/ICAO", "CUU")
    lat = st.number_input("Latitud", value=28.6353, format="%.4f")
    lon = st.number_input("Longitud", value=-106.0889, format="%.4f")
    hours = st.slider("Horas a analizar", 6, 72, 24)
    run = st.button("🚀 Ejecutar análisis", type="primary", use_container_width=True)
    st.info("Las API keys se colocan en el archivo .env. Si no hay llaves, se ejecuta en modo demo.")

if run:
    settings = Settings(city=city, airport=airport, lat=lat, lon=lon, hours=hours, output_dir=ROOT / "salidas")
    with st.spinner("Consultando APIs y procesando en paralelo..."):
        result = run_pipeline(settings, save=True)
    st.session_state["result"] = result

result = st.session_state.get("result")
if not result:
    st.markdown("""
    <div class="big-card">
    <h3>Sistema listo para ejecutar</h3>
    <p>Presiona <b>Ejecutar análisis</b>. El flujo consulta AirLabs, OpenWeather, OpenAQ y ThingSpeak con Asyncio, distribuye vuelos con MPI y calcula correlaciones con CUDA/CuPy o respaldo NumPy.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

metrics = result["metrics"]
corr = result["correlations"]
data = result["data"]
model = result["model"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Vuelos", f"{metrics.get('total_flights', 0):,.0f}")
c2.metric("Cancelados", f"{metrics.get('cancelled', 0):,.0f}")
c3.metric("Retraso promedio", f"{metrics.get('avg_delay_min', 0):.2f} min")
c4.metric("Velocidad promedio", f"{metrics.get('avg_speed_kmh', 0):.1f} km/h")
c5.metric("Aceleración", str(corr.get("mode", "N/D")))

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Resumen", "🛫 Vuelos", "🌦️ Ambiente", "⚡ CUDA/MPI", "📁 Evidencias"])

with tab1:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(x=list(metrics.get("airlines", {}).keys()), y=list(metrics.get("airlines", {}).values()), labels={"x":"Aerolínea", "y":"Vuelos"}, title="Vuelos por aerolínea"), use_container_width=True)
    with right:
        route_dict = metrics.get("routes", {})
        st.plotly_chart(px.bar(x=list(route_dict.values()), y=list(route_dict.keys()), orientation="h", labels={"x":"Vuelos", "y":"Ruta"}, title="Rutas más frecuentes"), use_container_width=True)
    st.plotly_chart(px.line(model, x="hour", y=["delay_min", "active_flights"], title="Retrasos y actividad aérea por hora"), use_container_width=True)

with tab2:
    flights = data["flights"]
    st.plotly_chart(px.scatter_mapbox(flights, lat="lat", lon="lon", color="airline", size="delay_min", hover_name="flight", zoom=4, height=520, mapbox_style="open-street-map", title="Mapa de vuelos y retrasos"), use_container_width=True)
    st.dataframe(flights, use_container_width=True, height=320)

with tab3:
    st.plotly_chart(px.scatter(model, x="rain_mm", y="delay_min", size="active_flights", hover_data=["visibility_m", "humidity_pct"], title="Lluvia vs retraso promedio"), use_container_width=True)
    cols = [c for c in ["temperature_c", "humidity_pct", "rain_mm", "visibility_m", "wind_kmh", "pm25", "pm10"] if c in model]
    st.plotly_chart(px.line(model, x="hour", y=cols, title="Variables ambientales integradas"), use_container_width=True)
    st.dataframe(model, use_container_width=True, height=300)

with tab4:
    st.subheader("Resultados de procesamiento paralelo")
    st.json({"metricas_mpi": metrics, "correlaciones_cuda": corr, "tiempos": result["timings"]})
    st.write("Atípicos detectados en velocidad, altitud o retraso:")
    st.dataframe(result["outliers"], use_container_width=True, height=280)

with tab5:
    st.success("Los archivos se guardaron en la carpeta salidas/.")
    st.code("mpiexec -n 4 python main.py --city Chihuahua --lat 28.6353 --lon -106.0889 --airport CUU --hours 24")
    st.code("streamlit run app/dashboard.py")
