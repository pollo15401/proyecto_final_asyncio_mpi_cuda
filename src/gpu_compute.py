"""Cálculo numérico acelerado con CUDA/CuPy y respaldo CPU."""
from typing import Dict, Tuple
import numpy as np
import pandas as pd

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

try:
    from numba import cuda
except Exception:  # pragma: no cover
    cuda = None


def _corr_cpu(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _corr_cupy(x: np.ndarray, y: np.ndarray) -> float:
    if cp is None:
        raise RuntimeError("CuPy no disponible")
    xg = cp.asarray(x, dtype=cp.float32)
    yg = cp.asarray(y, dtype=cp.float32)
    xm = xg - cp.mean(xg)
    ym = yg - cp.mean(yg)
    denom = cp.sqrt(cp.sum(xm ** 2) * cp.sum(ym ** 2))
    if float(cp.asnumpy(denom)) == 0.0:
        return 0.0
    return float(cp.asnumpy(cp.sum(xm * ym) / denom))


def build_model_table(weather: pd.DataFrame, flights: pd.DataFrame, air_quality: pd.DataFrame) -> pd.DataFrame:
    delays = flights.groupby("hour", as_index=False).agg(
        delay_min=("delay_min", "mean"),
        active_flights=("flight", "count"),
        avg_speed_kmh=("speed_kmh", "mean"),
        avg_altitude_m=("altitude_m", "mean"),
    )
    df = weather.merge(delays, on="hour", how="left").merge(air_quality, on="hour", how="left", suffixes=("", "_aq"))
    df["delay_min"] = df["delay_min"].fillna(0)
    df["active_flights"] = df["active_flights"].fillna(0)
    return df.fillna(method="ffill").fillna(method="bfill").fillna(0)


def correlations_gpu_or_cpu(model: pd.DataFrame) -> Dict[str, float | str]:
    features = ["rain_mm", "visibility_m", "humidity_pct", "wind_kmh", "pm25", "pm10", "active_flights"]
    target = model["delay_min"].to_numpy(dtype=np.float32)
    result: Dict[str, float | str] = {}
    try:
        for col in features:
            if col in model.columns:
                result[f"corr_{col}_delay"] = _corr_cupy(model[col].to_numpy(dtype=np.float32), target)
        result["mode"] = "GPU CuPy"
    except Exception:
        for col in features:
            if col in model.columns:
                result[f"corr_{col}_delay"] = _corr_cpu(model[col].to_numpy(dtype=np.float32), target)
        result["mode"] = "CPU NumPy fallback"
    return result


def detect_outliers_cuda_or_cpu(flights: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    values = flights[["speed_kmh", "altitude_m", "delay_min"]].to_numpy(dtype=np.float32)
    if cp is not None:
        try:
            vg = cp.asarray(values)
            mean = cp.mean(vg, axis=0)
            std = cp.std(vg, axis=0) + 1e-6
            z = cp.abs((vg - mean) / std)
            mask = cp.asnumpy(cp.any(z > 2.5, axis=1))
            out = flights.loc[mask].copy()
            out["outlier_reason"] = "z-score CUDA > 2.5"
            return out, "GPU CuPy"
        except Exception:
            pass
    mean = values.mean(axis=0)
    std = values.std(axis=0) + 1e-6
    mask = np.any(np.abs((values - mean) / std) > 2.5, axis=1)
    out = flights.loc[mask].copy()
    out["outlier_reason"] = "z-score CPU > 2.5"
    return out, "CPU NumPy fallback"
