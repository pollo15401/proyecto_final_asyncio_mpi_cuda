from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd


def save_charts(output: Path, flights: pd.DataFrame, model: pd.DataFrame, metrics: Dict) -> None:
    graph_dir = output / "graficas"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # --- Vuelos por aerolínea ---
    if not flights.empty and "airline" in flights.columns:
        plt.figure(figsize=(8, 4.5))
        flights["airline"].value_counts().plot(kind="bar")
        plt.title("Vuelos por aerolínea")
        plt.xlabel("Aerolínea")
        plt.ylabel("Cantidad")
        plt.tight_layout()
        plt.savefig(graph_dir / "vuelos_por_aerolinea.png", dpi=180)
        plt.close()

    # --- Retraso promedio por hora ---
    if not flights.empty and "hour" in flights.columns and "delay_min" in flights.columns:
        plt.figure(figsize=(8, 4.5))
        flights.groupby("hour")["delay_min"].mean().plot(marker="o")
        plt.title("Retraso promedio por hora")
        plt.xlabel("Hora")
        plt.ylabel("Retraso promedio (min)")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(graph_dir / "retraso_por_hora.png", dpi=180)
        plt.close()

    # --- Lluvia vs retraso ---
    if not model.empty and "rain_mm" in model.columns and "delay_min" in model.columns:
        plt.figure(figsize=(8, 4.5))
        plt.scatter(model["rain_mm"], model["delay_min"], alpha=0.75)
        plt.title("Relación lluvia vs retraso")
        plt.xlabel("Lluvia (mm)")
        plt.ylabel("Retraso promedio (min)")
        plt.tight_layout()
        plt.savefig(graph_dir / "lluvia_vs_retraso.png", dpi=180)
        plt.close()

    # --- Rutas más frecuentes ---
    route_series = pd.Series(metrics.get("routes", {}))
    plt.figure(figsize=(8, 4.5))
    if not route_series.empty:
        route_series.plot(kind="barh")
    plt.title("Rutas más frecuentes")
    plt.xlabel("Cantidad de vuelos")
    plt.tight_layout()
    plt.savefig(graph_dir / "rutas_frecuentes.png", dpi=180)
    plt.close()
