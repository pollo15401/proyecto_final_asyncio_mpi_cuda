"""Distribución del procesamiento con MPI4Py."""
from typing import Dict, List
import pandas as pd

try:
    from mpi4py import MPI
except Exception:  # pragma: no cover
    MPI = None


def is_mpi() -> bool:
    return MPI is not None


def rank_size() -> tuple[int, int]:
    if MPI is None:
        return 0, 1
    comm = MPI.COMM_WORLD
    return comm.Get_rank(), comm.Get_size()


def broadcast_data(data):
    if MPI is None:
        return data
    return MPI.COMM_WORLD.bcast(data, root=0)


def scatter_flights(flights: pd.DataFrame) -> pd.DataFrame:
    if MPI is None:
        return flights
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    parts = [part.copy() for part in __import__("numpy").array_split(flights, size)] if rank == 0 else None
    return comm.scatter(parts, root=0)


def local_metrics(local: pd.DataFrame) -> Dict:
    if local.empty:
        return {"n": 0, "cancelled": 0, "delay_sum": 0, "speed_sum": 0, "alt_sum": 0, "airlines": {}, "routes": {}}
    routes = (local["origin"].astype(str) + "-" + local["destination"].astype(str)).value_counts().head(10).to_dict()
    return {
        "n": int(len(local)),
        "cancelled": int(local["cancelled"].sum()) if "cancelled" in local else 0,
        "delay_sum": float(local["delay_min"].sum()),
        "speed_sum": float(local["speed_kmh"].sum()),
        "alt_sum": float(local["altitude_m"].sum()),
        "airlines": local["airline"].value_counts().to_dict(),
        "routes": routes,
    }


def gather_metrics(local_result: Dict) -> Dict:
    if MPI is None:
        results = [local_result]
    else:
        comm = MPI.COMM_WORLD
        results = comm.gather(local_result, root=0)
        if comm.Get_rank() != 0:
            return {}
    n = sum(r["n"] for r in results)
    airlines: Dict[str, int] = {}
    routes: Dict[str, int] = {}
    for r in results:
        for k, v in r["airlines"].items():
            airlines[k] = airlines.get(k, 0) + int(v)
        for k, v in r["routes"].items():
            routes[k] = routes.get(k, 0) + int(v)
    routes = dict(sorted(routes.items(), key=lambda x: x[1], reverse=True)[:10])
    return {
        "total_flights": n,
        "cancelled": sum(r["cancelled"] for r in results),
        "avg_delay_min": sum(r["delay_sum"] for r in results) / max(n, 1),
        "avg_speed_kmh": sum(r["speed_sum"] for r in results) / max(n, 1),
        "avg_altitude_m": sum(r["alt_sum"] for r in results) / max(n, 1),
        "airlines": dict(sorted(airlines.items(), key=lambda x: x[1], reverse=True)),
        "routes": routes,
    }
