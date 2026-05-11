import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .api_clients import collect_all
from .config import Settings
from .gpu_compute import build_model_table, correlations_gpu_or_cpu, detect_outliers_cuda_or_cpu
from .mpi_processing import rank_size, broadcast_data, scatter_flights, local_metrics, gather_metrics
from .visualization import save_charts


def save_outputs(settings: Settings, data: Dict[str, pd.DataFrame], model: pd.DataFrame, metrics: Dict, correlations: Dict, outliers: pd.DataFrame, timings: Dict) -> None:
    settings.ensure_dirs()
    csv_dir = settings.output_dir / "csv"
    data["weather"].to_csv(csv_dir / "openweather_clima.csv", index=False)
    data["thingspeak"].to_csv(csv_dir / "thingspeak_sensores.csv", index=False)
    data["air_quality"].to_csv(csv_dir / "openaq_calidad_aire.csv", index=False)
    data["flights"].to_csv(csv_dir / "airlabs_vuelos.csv", index=False)
    model.to_csv(csv_dir / "tabla_modelo_integrada.csv", index=False)
    outliers.to_csv(csv_dir / "outliers_detectados.csv", index=False)
    save_charts(settings.output_dir, data["flights"], model, metrics)
    report = {"settings": settings.__dict__ | {"output_dir": str(settings.output_dir)}, "metrics": metrics, "correlations": correlations, "timings": timings}
    with open(settings.output_dir / "resumen_resultados.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    with open(settings.output_dir / "resumen_ejecucion.txt", "w", encoding="utf-8") as f:
        f.write("PROYECTO FINAL: ANÁLISIS DE TRÁFICO AÉREO Y AMBIENTE\n")
        f.write(f"Ciudad: {settings.city} | Aeropuerto: {settings.airport}\n")
        f.write(f"Vuelos procesados: {metrics.get('total_flights', 0)}\n")
        f.write(f"Cancelados: {metrics.get('cancelled', 0)}\n")
        f.write(f"Retraso promedio: {metrics.get('avg_delay_min', 0):.2f} min\n")
        f.write(f"Velocidad promedio: {metrics.get('avg_speed_kmh', 0):.2f} km/h\n")
        f.write(f"Altitud promedio: {metrics.get('avg_altitude_m', 0):.2f} m\n")
        f.write(f"Modo aceleración: {correlations.get('mode')}\n")
        for k, v in correlations.items():
            f.write(f"{k}: {v}\n")
        f.write("\nTiempos:\n")
        for k, v in timings.items():
            f.write(f"{k}: {v:.4f} segundos\n")


def run_pipeline(settings: Settings, save: bool = True) -> Dict[str, Any]:
    rank, size = rank_size()
    timings: Dict[str, float] = {}
    if rank == 0:
        print("[1/6] Asyncio: consultando AirLabs, OpenWeather, OpenAQ y ThingSpeak en paralelo...")
        t0 = time.perf_counter()
        data = asyncio.run(collect_all(settings))
        timings["asyncio_api_seconds"] = time.perf_counter() - t0
    else:
        data = None
        timings = {}

    data = broadcast_data(data)
    if rank == 0:
        print(f"[2/6] MPI: distribuyendo {len(data['flights'])} vuelos entre {size} proceso(s)...")
    t1 = time.perf_counter()
    local = scatter_flights(data["flights"])
    local_result = local_metrics(local)
    metrics = gather_metrics(local_result)
    if rank == 0:
        timings["mpi_processing_seconds"] = time.perf_counter() - t1
        print("[3/6] Pandas: integrando clima, calidad del aire, sensores y retrasos...")
        t2 = time.perf_counter()
        model = build_model_table(data["weather"], data["flights"], data["air_quality"])
        timings["pandas_integration_seconds"] = time.perf_counter() - t2
        print("[4/6] CUDA/CuPy: calculando correlaciones y detección de atípicos...")
        t3 = time.perf_counter()
        correlations = correlations_gpu_or_cpu(model)
        outliers, outlier_mode = detect_outliers_cuda_or_cpu(data["flights"])
        correlations["outlier_mode"] = outlier_mode
        timings["cuda_or_cpu_seconds"] = time.perf_counter() - t3
        if save:
            print("[5/6] Guardando CSV, JSON, gráficas y resumen...")
            save_outputs(settings, data, model, metrics, correlations, outliers, timings)
        print("[6/6] Ejecución terminada correctamente.")
        return {"data": data, "model": model, "metrics": metrics, "correlations": correlations, "outliers": outliers, "timings": timings}
    return {}
