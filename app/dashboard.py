import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from src.config import Settings
from src.pipeline import run_pipeline

st.set_page_config(page_title="Proyecto Final | Asyncio + MPI + CUDA", layout="wide", page_icon="✈️")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
.big-card {border-radius: 18px; padding: 18px; background: #0f172a; color: white;}
.question-box {
    border-left: 4px solid #3b82f6;
    background: #1e293b;
    color: #e2e8f0;
    padding: 10px 16px;
    border-radius: 0 10px 10px 0;
    margin-bottom: 12px;
    font-size: 1.05rem;
}
.answer-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px 16px;
    color: #94a3b8;
    margin-bottom: 18px;
    font-size: 0.97rem;
}
</style>
""", unsafe_allow_html=True)

st.title("✈️ Dashboard de Tráfico Aéreo y Condiciones Ambientales")
st.caption("Proyecto final con Asyncio, MPI4Py, CUDA/CuPy/Numba, Pandas, APIs en la nube y visualización interactiva.")


def add_trendline(fig: go.Figure, x_data, y_data, color: str = "red") -> go.Figure:
    """Línea de tendencia lineal con numpy — sin statsmodels ni scipy."""
    try:
        x = np.array(x_data, dtype=float)
        y = np.array(y_data, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return fig
        # Necesitamos varianza suficiente en x para que polyfit no falle
        if np.std(x[mask]) < 1e-9 or np.std(y[mask]) < 1e-9:
            return fig
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, b = np.polyfit(x[mask], y[mask], 1)
        if not np.isfinite(m) or not np.isfinite(b):
            return fig
        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
        y_line = m * x_line + b
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                 line=dict(color=color, width=2, dash="dash"),
                                 name=f"Tendencia (m={m:.3f})"))
    except Exception:
        pass
    return fig

with st.sidebar:
    st.header("Configuración")

    # Presets de ciudades mexicanas para facilitar el uso
    CITY_PRESETS = {
        "Chihuahua":       ("CUU", 28.6353, -106.0889),
        "Ciudad de México": ("MEX", 19.4363, -99.0721),
        "Guadalajara":     ("GDL", 20.5218, -103.3109),
        "Monterrey":       ("MTY", 25.7785, -100.1065),
        "Tijuana":         ("TIJ", 32.5411, -116.9700),
        "Cancún":          ("CUN", 21.0365, -86.8771),
        "Puerto Vallarta": ("PVR", 20.6801, -105.2544),
        "Los Cabos":       ("SJD", 23.1518, -109.7215),
        "Hermosillo":      ("HMO", 29.0958, -111.0478),
        "Personalizado":   (None, None, None),
    }

    preset = st.selectbox("Ciudad / Aeropuerto", list(CITY_PRESETS.keys()), index=0)
    preset_airport, preset_lat, preset_lon = CITY_PRESETS[preset]

    if preset == "Personalizado":
        city    = st.text_input("Nombre de ciudad", "Mi Ciudad")
        airport = st.text_input("Aeropuerto IATA (3 letras)", "CUU").upper().strip()
        lat     = st.number_input("Latitud", value=28.6353, format="%.4f")
        lon     = st.number_input("Longitud", value=-106.0889, format="%.4f")
    else:
        city    = preset
        airport = preset_airport
        lat     = preset_lat
        lon     = preset_lon
        st.info(f"✈️ Aeropuerto: **{airport}** | 📍 {lat:.4f}, {lon:.4f}")

    hours = st.slider("Horas a analizar", 6, 72, 24)

    st.divider()
    run = st.button("🚀 Ejecutar análisis", type="primary", use_container_width=True)
    st.caption("Las API keys se configuran en el archivo .env")

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

# Fuente y timestamp de los datos
flights_src = data["flights"]["source"].iloc[0] if "source" in data["flights"].columns else "AirLabs"
ts = pd.Timestamp.now().strftime("%d/%m/%Y %H:%M")
st.caption(f"📡 Datos obtenidos: {ts} | Fuente vuelos: AirLabs /schedules + /flights | "
           f"Clima: OpenWeather | Sensores: ThingSpeak canal {result.get('settings', {}).get('thingspeak_channel_id', '12397') if isinstance(result.get('settings'), dict) else '12397'}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Resumen", "🛫 Vuelos", "🌦️ Ambiente", "⚡ CUDA/MPI", "📁 Evidencias"])

with tab1:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(x=list(metrics.get("airlines", {}).keys()), y=list(metrics.get("airlines", {}).values()), labels={"x":"Aerolínea", "y":"Vuelos"}, title="Vuelos por aerolínea"), use_container_width=True)
    with right:
        route_dict = metrics.get("routes", {})
        st.plotly_chart(px.bar(x=list(route_dict.values()), y=list(route_dict.keys()), orientation="h", labels={"x":"Vuelos", "y":"Ruta"}, title="Rutas más frecuentes"), use_container_width=True)

    # Actividad aérea por hora del día
    st.subheader("🕐 Actividad aérea por hora del día")
    flights_tab1 = data["flights"]
    if not flights_tab1.empty and "hour" in flights_tab1.columns:
        hourly = flights_tab1.copy()
        hourly["hora_dia"] = pd.to_datetime(hourly["hour"]).dt.hour
        hourly_count = hourly.groupby("hora_dia").agg(
            vuelos=("flight", "count"),
            retraso_prom=("delay_min", "mean"),
            velocidad_prom=("speed_kmh", "mean"),
            altitud_prom=("altitude_m", "mean"),
        ).reset_index()
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            st.plotly_chart(px.bar(hourly_count, x="hora_dia", y="vuelos",
                                   labels={"hora_dia": "Hora del día", "vuelos": "Cantidad de vuelos"},
                                   title="Vuelos activos por hora del día",
                                   color="vuelos", color_continuous_scale="Blues"),
                            use_container_width=True)
        with col_act2:
            st.plotly_chart(px.line(hourly_count, x="hora_dia", y="retraso_prom",
                                    labels={"hora_dia": "Hora del día", "retraso_prom": "Retraso promedio (min)"},
                                    title="Retraso promedio por hora del día", markers=True),
                            use_container_width=True)

    # Retrasos por aeropuerto
    st.subheader("🛬 Retrasos por aeropuerto (origen)")
    airport_delays = metrics.get("airport_delays", {})
    airport_flights_m = metrics.get("airport_flights", {})
    if airport_delays:
        ap_df = pd.DataFrame({
            "aeropuerto": list(airport_delays.keys()),
            "retraso_prom_min": [round(v, 2) for v in airport_delays.values()],
            "vuelos": [airport_flights_m.get(k, 0) for k in airport_delays.keys()],
        }).sort_values("retraso_prom_min", ascending=False)
        col_ap1, col_ap2 = st.columns(2)
        with col_ap1:
            st.plotly_chart(px.bar(ap_df, x="aeropuerto", y="retraso_prom_min",
                                   color="retraso_prom_min", color_continuous_scale="Reds",
                                   labels={"aeropuerto": "Aeropuerto", "retraso_prom_min": "Retraso prom. (min)"},
                                   title="Retraso promedio por aeropuerto de origen"),
                            use_container_width=True)
        with col_ap2:
            st.plotly_chart(px.bar(ap_df, x="aeropuerto", y="vuelos",
                                   color="vuelos", color_continuous_scale="Greens",
                                   labels={"aeropuerto": "Aeropuerto", "vuelos": "Vuelos"},
                                   title="Vuelos por aeropuerto de origen"),
                            use_container_width=True)

    # Velocidad y altitud promedio
    st.subheader("✈️ Velocidad y altitud promedio")
    m1, m2, m3 = st.columns(3)
    m1.metric("Velocidad promedio", f"{metrics.get('avg_speed_kmh', 0):.1f} km/h")
    m2.metric("Altitud promedio", f"{metrics.get('avg_altitude_m', 0):.0f} m")
    m3.metric("Retraso promedio global", f"{metrics.get('avg_delay_min', 0):.2f} min")

    st.plotly_chart(px.line(model, x="hour", y=["delay_min", "active_flights"],
                            title="Retrasos y actividad aérea por hora",
                            labels={"value": "Valor", "variable": "Métrica", "hour": "Hora"}),
                    use_container_width=True)

with tab2:
    flights = data["flights"]
    # size_col debe ser > 0 para scatter_mapbox; usar delay+1 para evitar ceros
    flights_map = flights.copy()
    flights_map["size_col"] = (flights_map["delay_min"].clip(lower=0) + 1).fillna(1)
    st.plotly_chart(px.scatter_mapbox(flights_map, lat="lat", lon="lon", color="airline",
                                      size="size_col", hover_name="flight",
                                      hover_data={"delay_min": True, "status": True, "origin": True, "destination": True, "size_col": False},
                                      zoom=4, height=520, mapbox_style="open-street-map",
                                      title="Mapa de vuelos en tiempo real"),
                    use_container_width=True)

    # --- Consulta detallada de vuelo específico ---
    st.subheader("🔍 Consulta de vuelo específico")
    flight_ids = sorted(flights["flight"].dropna().unique().tolist())
    selected_flight = st.selectbox("Selecciona un vuelo", flight_ids)
    if selected_flight:
        frow = flights[flights["flight"] == selected_flight].iloc[0]
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Vuelo", str(frow["flight"]))
        fc2.metric("Aerolínea", str(frow["airline"]) if "airline" in frow.index else "N/D")
        fc3.metric("Origen → Destino", f"{frow['origin'] if 'origin' in frow.index else '?'} → {frow['destination'] if 'destination' in frow.index else '?'}")
        fc4.metric("Estado", str(frow["status"]) if "status" in frow.index else "N/D")
        fc5, fc6, fc7, fc8 = st.columns(4)
        fc5.metric("Velocidad", f"{float(frow['speed_kmh']):.0f} km/h" if "speed_kmh" in frow.index else "N/D")
        fc6.metric("Altitud", f"{float(frow['altitude_m']):.0f} m" if "altitude_m" in frow.index else "N/D")
        fc7.metric("Retraso", f"{float(frow['delay_min']):.1f} min" if "delay_min" in frow.index else "N/D")
        fc8.metric("Cancelado", "Sí" if bool(frow["cancelled"]) else "No")

    # --- Simulación de ruta en mapa ---
    st.subheader("🗺️ Simulación de ruta")
    col_orig, col_dest = st.columns(2)
    all_airports = sorted(set(flights["origin"].dropna().tolist() + flights["destination"].dropna().tolist()))
    sim_orig = col_orig.selectbox("Aeropuerto origen", all_airports, key="sim_orig")
    sim_dest = col_dest.selectbox("Aeropuerto destino", all_airports, index=min(1, len(all_airports)-1), key="sim_dest")
    route_flights = flights[(flights["origin"] == sim_orig) & (flights["destination"] == sim_dest)]
    if not route_flights.empty:
        st.caption(f"{len(route_flights)} vuelo(s) en esta ruta — mostrando trayectoria simulada")
        # Crear puntos intermedios simulados entre origen y destino
        orig_lat = route_flights["lat"].iloc[0]
        orig_lon = route_flights["lon"].iloc[0]
        dest_lat = route_flights["lat"].iloc[-1]
        dest_lon = route_flights["lon"].iloc[-1]
        n_pts = 20
        lats = np.linspace(orig_lat, dest_lat, n_pts) + np.random.default_rng(7).normal(0, 0.15, n_pts)
        lons = np.linspace(orig_lon, dest_lon, n_pts) + np.random.default_rng(7).normal(0, 0.15, n_pts)
        route_df = pd.DataFrame({"lat": lats, "lon": lons, "punto": [f"P{i}" for i in range(n_pts)]})
        fig_route = px.line_mapbox(route_df, lat="lat", lon="lon", hover_name="punto",
                                   zoom=3, height=400, mapbox_style="open-street-map",
                                   title=f"Ruta simulada {sim_orig} → {sim_dest}")
        fig_route.add_scattermapbox(lat=[orig_lat, dest_lat], lon=[orig_lon, dest_lon],
                                    mode="markers", marker=dict(size=14, color=["green", "red"]),
                                    name="Origen / Destino")
        st.plotly_chart(fig_route, use_container_width=True)
    else:
        st.info(f"No hay vuelos registrados en la ruta {sim_orig} → {sim_dest} en los datos actuales.")

    st.subheader("📋 Tabla de vuelos")
    st.dataframe(flights, use_container_width=True, height=320)

with tab3:
    st.subheader("🌧️ Relación entre clima y retrasos")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_rain = px.scatter(model, x="rain_mm", y="delay_min",
                              size="active_flights" if "active_flights" in model.columns else None,
                              hover_data=[c for c in ["visibility_m", "humidity_pct", "wind_kmh"] if c in model.columns],
                              title="Lluvia vs retraso promedio",
                              labels={"rain_mm": "Lluvia (mm)", "delay_min": "Retraso prom. (min)"})
        if len(model) > 3:
            fig_rain = add_trendline(fig_rain, model["rain_mm"], model["delay_min"])
        st.plotly_chart(fig_rain, use_container_width=True)
    with col_c2:
        if "visibility_m" in model.columns:
            fig_vis = px.scatter(model, x="visibility_m", y="delay_min",
                                 title="Visibilidad vs retraso",
                                 labels={"visibility_m": "Visibilidad (m)", "delay_min": "Retraso prom. (min)"},
                                 color="rain_mm" if "rain_mm" in model.columns else None)
            if len(model) > 3:
                fig_vis = add_trendline(fig_vis, model["visibility_m"], model["delay_min"])
            st.plotly_chart(fig_vis, use_container_width=True)

    # Condiciones ambientales que coinciden con mayor tráfico
    st.subheader("📊 Condiciones ambientales vs tráfico aéreo")
    env_cols = [c for c in ["temperature_c", "humidity_pct", "rain_mm", "wind_kmh", "pm25", "pm10"] if c in model.columns]
    if env_cols and "active_flights" in model.columns:
        env_sel = st.selectbox("Variable ambiental", env_cols, key="env_traffic")
        fig_env = px.scatter(model, x=env_sel, y="active_flights",
                             title=f"{env_sel} vs vuelos activos",
                             labels={env_sel: env_sel, "active_flights": "Vuelos activos"},
                             color="delay_min" if "delay_min" in model.columns else None,
                             color_continuous_scale="RdYlGn_r")
        if len(model) > 3:
            fig_env = add_trendline(fig_env, model[env_sel], model["active_flights"])
        st.plotly_chart(fig_env, use_container_width=True)

    # Variables ambientales integradas
    st.subheader("📈 Variables ambientales por hora")
    all_env = [c for c in ["temperature_c", "humidity_pct", "rain_mm", "visibility_m", "wind_kmh", "pm25", "pm10", "no2", "o3"] if c in model.columns]
    st.plotly_chart(px.line(model, x="hour", y=all_env,
                            title="Variables ambientales integradas (OpenWeather + OpenAQ)",
                            labels={"value": "Valor", "variable": "Variable", "hour": "Hora"}),
                    use_container_width=True)

    # ThingSpeak — todos los campos
    st.subheader("🌡️ Sensores ThingSpeak")
    ts_df = data["thingspeak"]
    ts_source = ts_df["source"].iloc[0] if "source" in ts_df.columns and not ts_df.empty else "demo"
    st.caption(f"Fuente: {ts_source} | {len(ts_df)} registros")
    ts_cols_map = {
        "field1_temp": "Temperatura (°C)",
        "field2_humidity": "Humedad (%)",
        "field3_pressure": "Presión (hPa)",
        "field4_pm25": "PM2.5 (µg/m³)",
        "field5_pm10": "PM10 (µg/m³)",
        "field6_light": "Nivel de luz (lux)",
        "field7_wind_speed": "Velocidad viento (km/h)",
        "field8_wind_dir": "Dirección viento (°)",
    }
    ts_available = [c for c in ts_cols_map if c in ts_df.columns]
    if ts_available and "hour" in ts_df.columns:
        ts_col1, ts_col2 = st.columns(2)
        half = len(ts_available) // 2
        for i, col in enumerate(ts_available):
            target_col = ts_col1 if i < half else ts_col2
            with target_col:
                st.plotly_chart(px.line(ts_df, x="hour", y=col,
                                        title=ts_cols_map[col],
                                        labels={"hour": "Hora", col: ts_cols_map[col]}),
                                use_container_width=True)
    # Rosa de vientos si hay dirección
    if "field8_wind_dir" in ts_df.columns and "field7_wind_speed" in ts_df.columns:
        st.subheader("🧭 Rosa de vientos (ThingSpeak)")
        wind_df = ts_df[["field7_wind_speed", "field8_wind_dir"]].dropna()
        if not wind_df.empty:
            fig_wind = go.Figure(go.Barpolar(
                r=wind_df["field7_wind_speed"].tolist(),
                theta=wind_df["field8_wind_dir"].tolist(),
                marker_color=wind_df["field7_wind_speed"].tolist(),
                marker_colorscale="Blues",
                opacity=0.8,
            ))
            fig_wind.update_layout(title="Rosa de vientos", polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig_wind, use_container_width=True)

    st.dataframe(model, use_container_width=True, height=300)

with tab4:
    st.subheader("⚡ Resultados de procesamiento paralelo")

    # Correlaciones CUDA/CPU
    st.markdown("**Correlaciones clima → retraso (calculadas con CUDA/CuPy o NumPy)**")
    corr_items = {k: v for k, v in corr.items() if k.startswith("corr_")}
    if corr_items:
        corr_df = pd.DataFrame({
            "Variable": [k.replace("corr_", "").replace("_delay", "") for k in corr_items],
            "Correlación con retraso": [round(float(v), 4) for v in corr_items.values()],
        }).sort_values("Correlación con retraso", key=abs, ascending=False)
        col_corr1, col_corr2 = st.columns(2)
        with col_corr1:
            st.plotly_chart(px.bar(corr_df, x="Variable", y="Correlación con retraso",
                                   color="Correlación con retraso",
                                   color_continuous_scale="RdBu",
                                   color_continuous_midpoint=0,
                                   title=f"Correlaciones ({corr.get('mode', 'N/D')})"),
                            use_container_width=True)
        with col_corr2:
            st.dataframe(corr_df, use_container_width=True)

    # Tiempos de ejecución
    st.markdown("**Tiempos de ejecución por etapa**")
    timings = result["timings"]
    if timings:
        t_df = pd.DataFrame({
            "Etapa": list(timings.keys()),
            "Segundos": [round(v, 4) for v in timings.values()],
        })
        st.plotly_chart(px.bar(t_df, x="Etapa", y="Segundos",
                               title="Tiempo por etapa (Asyncio / MPI / CUDA-CPU / Pandas)",
                               color="Segundos", color_continuous_scale="Viridis"),
                        use_container_width=True)

    # Atípicos
    st.markdown("**Detección de valores atípicos en velocidad, altitud y retraso (z-score > 2.5)**")
    outliers_df = result["outliers"]
    st.metric("Atípicos detectados", len(outliers_df))
    if not outliers_df.empty:
        st.dataframe(outliers_df, use_container_width=True, height=280)
    else:
        st.info("No se detectaron valores atípicos con los datos actuales.")

    st.markdown("**Métricas MPI completas**")
    st.json({"metricas_mpi": metrics, "correlaciones_cuda": corr})

with tab5:
    st.success("Los archivos se guardaron en la carpeta salidas/.")
    st.markdown("**Stack tecnológico utilizado:**")
    st.markdown("""
    | Tecnología | Uso |
    |---|---|
    | **Asyncio + aiohttp** | Lectura concurrente de AirLabs, OpenWeather, OpenAQ y ThingSpeak |
    | **MPI4Py** | Distribución del procesamiento de vuelos entre procesos |
    | **CUDA / CuPy / Numba** | Cálculo de correlaciones y detección de atípicos acelerado en GPU |
    | **Pandas** | Limpieza, integración y análisis tabular de datos |
    | **Matplotlib** | Generación de gráficas estáticas guardadas en salidas/graficas/ |
    | **Streamlit + Plotly** | Dashboard interactivo |
    """)
    st.markdown("**Comandos de ejecución:**")
    st.code("python main.py --city Chihuahua --lat 28.6353 --lon -106.0889 --airport CUU --hours 24")
    st.code("mpiexec -n 4 python main.py --city Chihuahua --lat 28.6353 --lon -106.0889 --airport CUU --hours 24")
    st.code(".venv\\Scripts\\streamlit run app/dashboard.py")
    st.markdown("**Archivos generados:**")
    output_dir = ROOT / "salidas"
    csv_files = list((output_dir / "csv").glob("*.csv")) if (output_dir / "csv").exists() else []
    img_files = list((output_dir / "graficas").glob("*.png")) if (output_dir / "graficas").exists() else []
    st.write(f"- {len(csv_files)} archivos CSV en salidas/csv/")
    st.write(f"- {len(img_files)} gráficas en salidas/graficas/")
    for f in csv_files:
        st.write(f"  📄 {f.name}")
    if img_files:
        st.markdown("**Gráficas generadas (Matplotlib):**")
        cols_img = st.columns(min(len(img_files), 2))
        for i, img in enumerate(img_files):
            cols_img[i % 2].image(str(img), caption=img.stem, width=600)
