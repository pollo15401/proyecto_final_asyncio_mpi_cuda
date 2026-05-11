"""Distribución del procesamiento con MPI4Py."""
from typing import Dict
import pandas as pd

try:
    from mpi4py import MPI
except Exception:
    MPI = None


def rank_size() -> tuple:
    if MPI is None:
        return 0, 1
    comm = MPI.COMM_WORLD
    return comm.Get_rank(), comm.Get_size()


def broadcast_data(data):
    return data if MPI is None else MPI.COMM_WORLD.bcast(data, root=0)


def scatter_flights(flights: pd.DataFrame) -> pd.DataFrame:
    if MPI is None:
        return flights
    import numpy as np
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    parts = [p.copy() for p in np.array_split(flights, size)] if rank == 0 else None
    return comm.scatter(parts, root=0)


def local_metrics(local: pd.DataFrame) -> Dict:
    empty = {"n": 0, "cancelled": 0, "delay_sum": 0.0, "speed_sum": 0.0, "speed_n": 0,
             "alt_sum": 0.0, "alt_n": 0, "airlines": {}, "routes": {},
             "airport_delays": {}, "airport_flights": {}}
    if local.empty:
        return empty

    routes = (
        local["origin"].astype(str) + "-" + local["destination"].astype(str)
    ).value_counts().head(10).to_dict()

    speed_data = local.loc[local["speed_kmh"] > 0, "speed_kmh"] if "speed_kmh" in local.columns else local["speed_kmh"].iloc[0:0]
    alt_data   = local.loc[local["altitude_m"] > 0, "altitude_m"] if "altitude_m" in local.columns else local["altitude_m"].iloc[0:0]

    airport_delays: Dict[str, float] = {}
    airport_flights: Dict[str, int] = {}
    for airport, grp in local.groupby("origin"):
        airport_delays[str(airport)] = float(grp["delay_min"].mean())
        airport_flights[str(airport)] = int(len(grp))

    return {
        "n":               int(len(local)),
        "cancelled":       int(local["cancelled"].sum()) if "cancelled" in local.columns else 0,
        "delay_sum":       float(local["delay_min"].sum()),
        "speed_sum":       float(speed_data.sum()),
        "speed_n":         int(len(speed_data)),
        "alt_sum":         float(alt_data.sum()),
        "alt_n":           int(len(alt_data)),
        "airlines":        local["airline"].value_counts().to_dict(),
        "routes":          routes,
        "airport_delays":  airport_delays,
        "airport_flights": airport_flights,
    }


def gather_metrics(local_result: Dict) -> Dict:
    if MPI is None:
        results = [local_result]
    else:
        comm = MPI.COMM_WORLD
        results = comm.gather(local_result, root=0)
        if comm.Get_rank() != 0:
            return {}

    n            = sum(r["n"] for r in results)
    speed_n      = sum(r.get("speed_n", 0) for r in results)
    alt_n        = sum(r.get("alt_n", 0) for r in results)
    airlines:        Dict[str, int]   = {}
    routes:          Dict[str, int]   = {}
    airport_delays:  Dict[str, float] = {}
    airport_flights: Dict[str, int]   = {}

    for r in results:
        for k, v in r["airlines"].items():
            airlines[k] = airlines.get(k, 0) + int(v)
        for k, v in r["routes"].items():
            routes[k] = routes.get(k, 0) + int(v)
        for k, v in r.get("airport_delays", {}).items():
            prev_n   = airport_flights.get(k, 0)
            new_n    = r.get("airport_flights", {}).get(k, 0)
            prev_sum = airport_delays.get(k, 0.0) * prev_n
            airport_flights[k] = prev_n + new_n
            airport_delays[k]  = (prev_sum + v * new_n) / max(prev_n + new_n, 1)

    return {
        "total_flights":  n,
        "cancelled":      sum(r["cancelled"] for r in results),
        "avg_delay_min":  sum(r["delay_sum"] for r in results) / max(n, 1),
        "avg_speed_kmh":  sum(r.get("speed_sum", 0) for r in results) / max(speed_n, 1),
        "avg_altitude_m": sum(r.get("alt_sum", 0) for r in results) / max(alt_n, 1),
        "airlines":       dict(sorted(airlines.items(), key=lambda x: x[1], reverse=True)),
        "routes":         dict(sorted(routes.items(), key=lambda x: x[1], reverse=True)[:10]),
        "airport_delays": dict(sorted(airport_delays.items(), key=lambda x: x[1], reverse=True)[:15]),
        "airport_flights": dict(sorted(airport_flights.items(), key=lambda x: x[1], reverse=True)[:15]),
    }


