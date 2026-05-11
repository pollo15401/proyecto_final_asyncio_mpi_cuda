"""
Clientes asíncronos para AirLabs, ThingSpeak, OpenWeather y OpenAQ.
Usa datos reales cuando hay API keys; cae a datos demo si falla cualquier fuente.
"""
import asyncio
import random
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]

import numpy as np
import pandas as pd

from .config import Settings

DEFAULT_HEADERS = {"User-Agent": "Proyecto-Final-Asyncio-MPI-CUDA/1.0"}


class APIClient:
    def __init__(self, timeout: int = 30) -> None:
        if aiohttp is None:
            raise RuntimeError("aiohttp no instalado.")
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def get_json(
        self,
        session: Any,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        async with session.get(url, params=params, headers=headers or DEFAULT_HEADERS) as r:
            r.raise_for_status()
            return await r.json()


def _hours(hours: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp.now().floor("h"), periods=hours, freq="h")


def demo_flights(settings: Settings, n: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    hours = _hours(settings.hours)
    airports = np.array([settings.airport, "MEX", "GDL", "MTY", "TIJ", "CJS", "HMO"])
    weather_factor = rng.choice([0, 1], size=n, p=[0.75, 0.25])
    delay = rng.gamma(2.0, 5.8, n) + weather_factor * rng.gamma(2.0, 8.0, n)
    df = pd.DataFrame({
        "flight":       [f"MX{1000 + i}" for i in range(n)],
        "hour":         rng.choice(hours, n),
        "airline":      rng.choice(["AM", "VB", "Y4", "AA", "UA", "DL", "AS"], n),
        "origin":       rng.choice(airports, n),
        "destination":  rng.choice(airports, n),
        "status":       rng.choice(["active", "scheduled", "landed", "cancelled"], n, p=[0.35, 0.35, 0.25, 0.05]),
        "speed_kmh":    rng.normal(725, 95, n).clip(180, 980),
        "altitude_m":   rng.normal(9000, 1700, n).clip(700, 12500),
        "delay_min":    delay.clip(0, 180),
        "duration_min": rng.normal(120, 40, n).clip(30, 360),
        "lat":          settings.lat + rng.normal(0, 2.2, n),
        "lon":          settings.lon + rng.normal(0, 2.2, n),
    })
    df["cancelled"] = df["status"].eq("cancelled")
    df["hour"] = pd.to_datetime(df["hour"]).dt.floor("h")
    return df


def demo_weather(settings: Settings) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    hours = _hours(settings.hours)
    rain = rng.exponential(0.22, settings.hours).clip(0, 12)
    return pd.DataFrame({
        "hour":          hours,
        "temperature_c": rng.normal(24, 6, settings.hours).round(2),
        "humidity_pct":  rng.normal(45, 17, settings.hours).clip(8, 100).round(2),
        "pressure_hpa":  rng.normal(1012, 6, settings.hours).round(2),
        "rain_mm":       rain.round(2),
        "visibility_m":  (11000 - rain * 900 + rng.normal(0, 900, settings.hours)).clip(400, 12000).round(0),
        "wind_kmh":      rng.normal(17, 7, settings.hours).clip(0, 70).round(2),
        "source":        "demo",
    })


def demo_air_quality(settings: Settings) -> pd.DataFrame:
    rng = np.random.default_rng(77)
    hours = _hours(settings.hours)
    return pd.DataFrame({
        "hour":   hours,
        "pm25":   rng.normal(18, 6, settings.hours).clip(1, 85).round(2),
        "pm10":   rng.normal(42, 14, settings.hours).clip(3, 160).round(2),
        "no2":    rng.normal(14, 5, settings.hours).clip(1, 60).round(2),
        "o3":     rng.normal(35, 11, settings.hours).clip(2, 130).round(2),
        "source": "demo",
    })


def demo_thingspeak(settings: Settings) -> pd.DataFrame:
    rng = np.random.default_rng(55)
    hours = _hours(settings.hours)
    return pd.DataFrame({
        "hour":              hours,
        "field1_temp":       rng.normal(25, 5, settings.hours).round(2),
        "field2_humidity":   rng.normal(45, 14, settings.hours).clip(0, 100).round(2),
        "field3_pressure":   rng.normal(1010, 7, settings.hours).round(2),
        "field4_pm25":       rng.normal(16, 5, settings.hours).clip(1, 80).round(2),
        "field5_pm10":       rng.normal(35, 12, settings.hours).clip(3, 150).round(2),
        "field6_light":      rng.uniform(0, 1000, settings.hours).round(1),
        "field7_wind_speed": rng.normal(15, 6, settings.hours).clip(0, 60).round(2),
        "field8_wind_dir":   rng.uniform(0, 360, settings.hours).round(1),
        "source":            "demo",
    })


async def fetch_openweather(settings: Settings, client: APIClient, session: Any) -> pd.DataFrame:
    if not settings.openweather_api_key:
        return demo_weather(settings)
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": settings.lat, "lon": settings.lon, "appid": settings.openweather_api_key, "units": "metric", "lang": "es"}
    try:
        data = await client.get_json(session, url, params=params)
        rows = []
        for item in data.get("list", [])[:settings.hours]:
            rows.append({
                "hour": pd.to_datetime(item.get("dt_txt")).floor("h"),
                "temperature_c": item.get("main", {}).get("temp"),
                "humidity_pct": item.get("main", {}).get("humidity"),
                "pressure_hpa": item.get("main", {}).get("pressure"),
                "rain_mm": item.get("rain", {}).get("3h", 0),
                "visibility_m": item.get("visibility", np.nan),
                "wind_kmh": item.get("wind", {}).get("speed", 0) * 3.6,
                "source": "OpenWeather",
            })
        return pd.DataFrame(rows) if rows else demo_weather(settings)
    except Exception as exc:
        print(f"[WARN] OpenWeather falló: {exc}. Usando demo.")
        return demo_weather(settings)


async def fetch_thingspeak(settings: Settings, client: APIClient, session: Any) -> pd.DataFrame:
    """Consulta ThingSpeak. Canal 12397 = WeatherStation pública con datos reales.
    Campos: field1=luz, field2=temp(°C aprox), field3=?, field4=lluvia, field6=presión.
    """
    url = f"https://api.thingspeak.com/channels/{settings.thingspeak_channel_id}/feeds.json"
    params: Dict[str, Any] = {"results": min(settings.hours * 2, 8000)}
    if settings.thingspeak_read_api_key:
        params["api_key"] = settings.thingspeak_read_api_key
    try:
        data = await client.get_json(session, url, params=params)
        feeds = data.get("feeds", [])
        if not feeds:
            print("[INFO] ThingSpeak: sin datos. Usando demo.")
            return demo_thingspeak(settings)
        rows = []
        for f in feeds:
            rows.append({
                "hour": pd.to_datetime(f.get("created_at")).floor("h"),
                "field1_temp": pd.to_numeric(f.get("field2"), errors="coerce"),
                "field2_humidity": pd.to_numeric(f.get("field3"), errors="coerce"),
                "field3_pressure": pd.to_numeric(f.get("field6"), errors="coerce"),
                "field4_pm25": pd.to_numeric(f.get("field4"), errors="coerce"),
                "field5_pm10": pd.to_numeric(f.get("field5"), errors="coerce"),
                "field6_light": pd.to_numeric(f.get("field1"), errors="coerce"),
                "field7_wind_speed": pd.to_numeric(f.get("field7"), errors="coerce"),
                "field8_wind_dir": pd.to_numeric(f.get("field8"), errors="coerce"),
                "source": "ThingSpeak",
            })
        df = pd.DataFrame(rows)
        df = df.groupby("hour", as_index=False).mean(numeric_only=True)
        df["source"] = "ThingSpeak"
        df = df.sort_values("hour").tail(settings.hours).reset_index(drop=True)
        df = df.dropna(how="all", subset=["field1_temp", "field2_humidity", "field3_pressure",
                                           "field4_pm25", "field5_pm10", "field6_light",
                                           "field7_wind_speed", "field8_wind_dir"])
        print(f"[OK] ThingSpeak: {len(df)} registros reales obtenidos.")
        return df if not df.empty else demo_thingspeak(settings)
    except Exception as exc:
        print(f"[WARN] ThingSpeak falló: {exc}. Usando demo.")
        return demo_thingspeak(settings)


async def fetch_openaq(settings: Settings, client: APIClient, session: Any) -> pd.DataFrame:
    """Consulta OpenAQ v3 /v2/measurements (compatible con clave gratuita).
    Si la API falla o no hay datos, usa datos demo.
    """
    # OpenAQ v3 requiere API key; si no hay, usamos demo directamente
    if not settings.openaq_api_key or settings.openaq_api_key.startswith("TU_"):
        return demo_air_quality(settings)

    url = "https://api.openaq.org/v3/locations"
    headers = DEFAULT_HEADERS.copy()
    headers["X-API-Key"] = settings.openaq_api_key
    params = {
        "coordinates": f"{settings.lat},{settings.lon}",
        "radius": 75000,
        "limit": 5,
    }
    try:
        data = await client.get_json(session, url, params=params, headers=headers)
        results = data.get("results", [])
        if not results:
            print("[INFO] OpenAQ: sin estaciones cercanas. Usando demo.")
            return demo_air_quality(settings)

        # Tomar el primer location_id y consultar sus mediciones recientes
        location_id = results[0].get("id")
        meas_url = f"https://api.openaq.org/v3/locations/{location_id}/measurements"
        meas_params = {"limit": settings.hours, "order_by": "datetime", "sort_order": "desc"}
        meas_data = await client.get_json(session, meas_url, params=meas_params, headers=headers)
        meas_results = meas_data.get("results", [])
        if not meas_results:
            return demo_air_quality(settings)

        # Agrupar por hora y parámetro
        rows_map: dict = {}
        for m in meas_results:
            dt = pd.to_datetime(m.get("period", {}).get("datetimeTo", {}).get("local") or
                                m.get("date", {}).get("local")).floor("h")
            param = m.get("parameter", {}).get("name", "").lower()
            value = m.get("value")
            if dt not in rows_map:
                rows_map[dt] = {"hour": dt, "source": "OpenAQ"}
            if param in ("pm25", "pm10", "no2", "o3") and value is not None:
                rows_map[dt][param] = float(value)

        df = pd.DataFrame(list(rows_map.values()))
        for col in ["pm25", "pm10", "no2", "o3"]:
            if col not in df.columns:
                df[col] = np.nan
        df = df.sort_values("hour").reset_index(drop=True)
        return df if not df.empty else demo_air_quality(settings)

    except Exception as exc:
        print(f"[WARN] OpenAQ falló: {exc}. Usando demo.")
        return demo_air_quality(settings)


async def fetch_airlabs(settings: Settings, client: APIClient, session: Any) -> pd.DataFrame:
    """Consulta AirLabs v9 /schedules para obtener vuelos con retrasos, cancelaciones
    y horarios reales. Filtra por aeropuertos mexicanos relevantes.
    Si no hay API key o la API falla, usa datos demo.
    """
    if not settings.airlabs_api_key:
        print("[INFO] AirLabs: sin API key. Usando demo.")
        return demo_flights(settings)

    # El aeropuerto configurado es el principal; los demás son los más cercanos geográficamente
    airport_iata = settings.airport if len(settings.airport) == 3 else "MEX"

    # Mapa de aeropuertos mexicanos con coordenadas para seleccionar los más cercanos
    MX_AIRPORTS_COORDS = {
        "MEX": (19.4363, -99.0721), "GDL": (20.5218, -103.3109), "MTY": (25.7785, -100.1065),
        "CUU": (28.6353, -106.0889), "TIJ": (32.5411, -116.9700), "CJS": (31.5361, -106.4296),
        "HMO": (29.0958, -111.0478), "MZT": (23.1614, -106.2661), "PVR": (20.6801, -105.2544),
        "CUN": (21.0365, -86.8771), "SJD": (23.1518, -109.7215), "AGU": (21.7056, -102.3181),
        "BJX": (20.9935, -101.4808), "OAX": (17.0000, -96.7266), "VSA": (17.9970, -92.8174),
        "MID": (20.9370, -89.6577), "TAM": (22.2964, -97.8659), "ZIH": (17.6016, -101.4605),
        "LAP": (24.0727, -110.3613), "ZLO": (19.1448, -104.5588),
    }

    # Ordenar por distancia a las coordenadas configuradas y tomar los 9 más cercanos
    import math
    def _dist(coords):
        dlat = coords[0] - settings.lat
        dlon = coords[1] - settings.lon
        return math.sqrt(dlat ** 2 + dlon ** 2)

    nearby = sorted(
        [(ap, c) for ap, c in MX_AIRPORTS_COORDS.items() if ap != airport_iata],
        key=lambda x: _dist(x[1])
    )
    query_airports = list(dict.fromkeys([airport_iata] + [ap for ap, _ in nearby[:9]]))

    url = "https://airlabs.co/api/v9/schedules"
    all_rows: List[Dict[str, Any]] = []

    try:
        for ap in query_airports:
            # Consultar salidas (dep) y llegadas (arr) para cada aeropuerto
            for direction in ("dep_iata", "arr_iata"):
                params = {
                    "api_key": settings.airlabs_api_key,
                    direction: ap,
                    "limit": 100,
                }
                try:
                    data = await client.get_json(session, url, params=params)
                    if "error" in data:
                        continue
                    for f in data.get("response", []):
                        delay = float(f.get("dep_delayed") or f.get("arr_delayed") or f.get("delayed") or 0)
                        status = str(f.get("status", "scheduled")).lower()
                        cancelled = status in ("cancelled", "canceled", "diverted")
                        dep_ts = f.get("dep_actual_utc") or f.get("dep_estimated_utc") or f.get("dep_time_utc")
                        try:
                            hour = pd.to_datetime(dep_ts).floor("h")
                        except Exception:
                            hour = pd.Timestamp.now().floor("h")
                        all_rows.append({
                            "flight":       f.get("flight_iata") or f.get("flight_icao") or "ND",
                            "hour":         hour,
                            "airline":      f.get("airline_iata") or f.get("airline_icao") or "ND",
                            "origin":       f.get("dep_iata") or f.get("dep_icao") or ap,
                            "destination":  f.get("arr_iata") or f.get("arr_icao") or "ND",
                            "status":       status,
                            "speed_kmh":    0.0,
                            "altitude_m":   0.0,
                            "delay_min":    delay,
                            "lat":          settings.lat + random.uniform(-3, 3),
                            "lon":          settings.lon + random.uniform(-3, 3),
                            "cancelled":    cancelled,
                            "duration_min": float(f.get("duration") or 0),
                        })
                except Exception:
                    continue

        if not all_rows:
            print("[INFO] AirLabs schedules: sin datos. Usando demo.")
            return demo_flights(settings)

        df = pd.DataFrame(all_rows)
        df["hour"] = pd.to_datetime(df["hour"]).dt.floor("h")
        df = df.drop_duplicates(subset=["flight", "hour"])

        # Enriquecer con velocidad/altitud de /flights para los vuelos en-route
        try:
            fl_params = {"api_key": settings.airlabs_api_key}
            fl_data = await client.get_json(session, "https://airlabs.co/api/v9/flights", params=fl_params)
            if "error" not in fl_data:
                live = [
                    {
                        "flight": f.get("flight_iata") or f.get("flight_icao"),
                        "speed_kmh_live": float(f.get("speed", 0) or 0),
                        "altitude_m_live": float(f.get("alt", 0) or 0) * 0.3048,
                        "lat_live": float(f.get("lat", settings.lat) or settings.lat),
                        "lon_live": float(f.get("lng", settings.lon) or settings.lon),
                    }
                    for f in fl_data.get("response", [])
                    if (f.get("flag") == "MX" or (
                        f.get("lat") and abs(float(f["lat"]) - settings.lat) <= 8
                    )) and (f.get("flight_iata") or f.get("flight_icao"))
                ]
                if live:
                    live_df = pd.DataFrame(live).drop_duplicates(subset=["flight"])
                    df = df.merge(live_df, on="flight", how="left")
                    mask = df["speed_kmh_live"].notna() & (df["speed_kmh_live"] > 0)
                    df.loc[mask, "speed_kmh"] = df.loc[mask, "speed_kmh_live"]
                    df.loc[mask, "altitude_m"] = df.loc[mask, "altitude_m_live"]
                    df.loc[mask, "lat"] = df.loc[mask, "lat_live"]
                    df.loc[mask, "lon"] = df.loc[mask, "lon_live"]
                    df = df.drop(columns=["speed_kmh_live", "altitude_m_live", "lat_live", "lon_live"])
        except Exception:
            pass

        print(f"[OK] AirLabs: {len(df)} vuelos reales (schedules) | "
              f"cancelados={df['cancelled'].sum()} | "
              f"retraso_prom={df['delay_min'].mean():.1f} min")
        return df

    except Exception as exc:
        print(f"[WARN] AirLabs falló: {exc}. Usando demo.")
        return demo_flights(settings)


async def collect_all(settings: Settings) -> Dict[str, pd.DataFrame]:
    """Etapa Asyncio: todas las APIs se consultan al mismo tiempo."""
    if aiohttp is None:
        print("[WARN] aiohttp no está instalado. Usando datos demo para todas las fuentes.")
        return {
            "weather": demo_weather(settings),
            "thingspeak": demo_thingspeak(settings),
            "air_quality": demo_air_quality(settings),
            "flights": demo_flights(settings),
        }

    client = APIClient()
    async with aiohttp.ClientSession(timeout=client.timeout) as session:
        weather, thingspeak, air_quality, flights = await asyncio.gather(
            fetch_openweather(settings, client, session),
            fetch_thingspeak(settings, client, session),
            fetch_openaq(settings, client, session),
            fetch_airlabs(settings, client, session),
        )
    return {"weather": weather, "thingspeak": thingspeak, "air_quality": air_quality, "flights": flights}
