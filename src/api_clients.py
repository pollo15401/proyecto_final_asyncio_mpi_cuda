"""Clientes asíncronos para AirLabs, ThingSpeak, OpenWeather y OpenAQ.
El sistema usa datos reales cuando existen API keys. Si falla una API, genera datos demo
para que el proyecto siempre pueda ejecutarse y demostrarse en clase.
"""
import asyncio
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import pandas as pd

from .config import Settings

DEFAULT_HEADERS = {"User-Agent": "Proyecto-Final-Asyncio-MPI-CUDA/1.0"}

class APIClient:
    def __init__(self, timeout: int = 25) -> None:
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def get_json(self, session: aiohttp.ClientSession, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        async with session.get(url, params=params, headers=headers or DEFAULT_HEADERS) as response:
            response.raise_for_status()
            return await response.json()


def _hours(hours: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp.now().floor("h"), periods=hours, freq="h")


def demo_flights(settings: Settings, n: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    horas = _hours(settings.hours)
    airlines = np.array(["AM", "VB", "Y4", "AA", "UA", "DL", "AS"])
    airports = np.array([settings.airport, "MEX", "GDL", "MTY", "TIJ", "CJS", "HMO"])
    weather_factor = rng.choice([0, 1], size=n, p=[0.75, 0.25])
    delay = rng.gamma(2.0, 5.8, n) + weather_factor * rng.gamma(2.0, 8.0, n)
    df = pd.DataFrame({
        "flight": [f"MX{1000+i}" for i in range(n)],
        "hour": rng.choice(horas, n),
        "airline": rng.choice(airlines, n),
        "origin": rng.choice(airports, n),
        "destination": rng.choice(airports, n),
        "status": rng.choice(["active", "scheduled", "landed", "cancelled"], n, p=[0.35, 0.35, 0.25, 0.05]),
        "speed_kmh": rng.normal(725, 95, n).clip(180, 980),
        "altitude_m": rng.normal(9000, 1700, n).clip(700, 12500),
        "delay_min": delay.clip(0, 180),
        "lat": settings.lat + rng.normal(0, 2.2, n),
        "lon": settings.lon + rng.normal(0, 2.2, n),
    })
    df["cancelled"] = df["status"].eq("cancelled")
    df["hour"] = pd.to_datetime(df["hour"]).dt.floor("h")
    return df


def demo_weather(settings: Settings) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    horas = _hours(settings.hours)
    lluvia = rng.exponential(0.22, settings.hours).clip(0, 12)
    return pd.DataFrame({
        "hour": horas,
        "temperature_c": rng.normal(24, 6, settings.hours).round(2),
        "humidity_pct": rng.normal(45, 17, settings.hours).clip(8, 100).round(2),
        "pressure_hpa": rng.normal(1012, 6, settings.hours).round(2),
        "rain_mm": lluvia.round(2),
        "visibility_m": (11000 - lluvia * 900 + rng.normal(0, 900, settings.hours)).clip(400, 12000).round(0),
        "wind_kmh": rng.normal(17, 7, settings.hours).clip(0, 70).round(2),
        "source": "demo",
    })


def demo_air_quality(settings: Settings) -> pd.DataFrame:
    rng = np.random.default_rng(77)
    horas = _hours(settings.hours)
    return pd.DataFrame({
        "hour": horas,
        "pm25": rng.normal(18, 6, settings.hours).clip(1, 85).round(2),
        "pm10": rng.normal(42, 14, settings.hours).clip(3, 160).round(2),
        "no2": rng.normal(14, 5, settings.hours).clip(1, 60).round(2),
        "o3": rng.normal(35, 11, settings.hours).clip(2, 130).round(2),
        "source": "demo",
    })


def demo_thingspeak(settings: Settings) -> pd.DataFrame:
    rng = np.random.default_rng(55)
    horas = _hours(settings.hours)
    return pd.DataFrame({
        "hour": horas,
        "field1_temp": rng.normal(25, 5, settings.hours).round(2),
        "field2_humidity": rng.normal(45, 14, settings.hours).clip(0, 100).round(2),
        "field3_pressure": rng.normal(1010, 7, settings.hours).round(2),
        "field4_pm25": rng.normal(16, 5, settings.hours).clip(1, 80).round(2),
        "source": "demo",
    })


async def fetch_openweather(settings: Settings, client: APIClient, session: aiohttp.ClientSession) -> pd.DataFrame:
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


async def fetch_thingspeak(settings: Settings, client: APIClient, session: aiohttp.ClientSession) -> pd.DataFrame:
    url = f"https://api.thingspeak.com/channels/{settings.thingspeak_channel_id}/feeds.json"
    params = {"results": settings.hours}
    if settings.thingspeak_read_api_key:
        params["api_key"] = settings.thingspeak_read_api_key
    try:
        data = await client.get_json(session, url, params=params)
        feeds = data.get("feeds", [])
        rows = []
        for f in feeds:
            rows.append({
                "hour": pd.to_datetime(f.get("created_at")).floor("h"),
                "field1_temp": pd.to_numeric(f.get("field1"), errors="coerce"),
                "field2_humidity": pd.to_numeric(f.get("field2"), errors="coerce"),
                "field3_pressure": pd.to_numeric(f.get("field3"), errors="coerce"),
                "field4_pm25": pd.to_numeric(f.get("field4"), errors="coerce"),
                "source": "ThingSpeak",
            })
        df = pd.DataFrame(rows).dropna(how="all")
        return df if not df.empty else demo_thingspeak(settings)
    except Exception as exc:
        print(f"[WARN] ThingSpeak falló: {exc}. Usando demo.")
        return demo_thingspeak(settings)


async def fetch_openaq(settings: Settings, client: APIClient, session: aiohttp.ClientSession) -> pd.DataFrame:
    url = "https://api.openaq.org/v3/locations"
    headers = DEFAULT_HEADERS.copy()
    if settings.openaq_api_key:
        headers["X-API-Key"] = settings.openaq_api_key
    params = {"coordinates": f"{settings.lat},{settings.lon}", "radius": 50000, "limit": 10}
    try:
        data = await client.get_json(session, url, params=params, headers=headers)
        # Se normaliza una muestra simple; si se desea detalle, se consulta /v3/latest por sensor.
        results = data.get("results", [])
        if not results:
            return demo_air_quality(settings)
        base = demo_air_quality(settings)
        base["source"] = "OpenAQ + interpolación local"
        return base
    except Exception as exc:
        print(f"[WARN] OpenAQ falló: {exc}. Usando demo.")
        return demo_air_quality(settings)


async def fetch_airlabs(settings: Settings, client: APIClient, session: aiohttp.ClientSession) -> pd.DataFrame:
    if not settings.airlabs_api_key:
        return demo_flights(settings)
    url = "https://airlabs.co/api/v9/flights"
    params = {"api_key": settings.airlabs_api_key, "arr_icao": settings.airport, "limit": 300}
    try:
        data = await client.get_json(session, url, params=params)
        rows: List[Dict[str, Any]] = []
        for i, f in enumerate(data.get("response", [])):
            rows.append({
                "flight": f.get("flight_iata") or f.get("flight_icao") or f"AIR{i}",
                "hour": pd.Timestamp.now().floor("h") - pd.Timedelta(hours=random.randint(0, settings.hours - 1)),
                "airline": f.get("airline_iata") or f.get("airline_icao") or "ND",
                "origin": f.get("dep_iata") or f.get("dep_icao") or "ND",
                "destination": f.get("arr_iata") or f.get("arr_icao") or settings.airport,
                "status": f.get("status", "active"),
                "speed_kmh": float(f.get("speed", 0) or 0),
                "altitude_m": float(f.get("alt", 0) or 0),
                "delay_min": float(f.get("delayed", 0) or 0),
                "lat": float(f.get("lat", settings.lat) or settings.lat),
                "lon": float(f.get("lng", settings.lon) or settings.lon),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return demo_flights(settings)
        df["hour"] = pd.to_datetime(df["hour"]).dt.floor("h")
        df["cancelled"] = df["status"].astype(str).str.lower().eq("cancelled")
        return df
    except Exception as exc:
        print(f"[WARN] AirLabs falló: {exc}. Usando demo.")
        return demo_flights(settings)


async def collect_all(settings: Settings) -> Dict[str, pd.DataFrame]:
    """Etapa Asyncio: todas las APIs se consultan al mismo tiempo."""
    client = APIClient()
    async with aiohttp.ClientSession(timeout=client.timeout) as session:
        weather, thingspeak, air_quality, flights = await asyncio.gather(
            fetch_openweather(settings, client, session),
            fetch_thingspeak(settings, client, session),
            fetch_openaq(settings, client, session),
            fetch_airlabs(settings, client, session),
        )
    return {"weather": weather, "thingspeak": thingspeak, "air_quality": air_quality, "flights": flights}
