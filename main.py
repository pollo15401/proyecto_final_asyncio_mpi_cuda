"""Entrada por consola del proyecto final.
Ejemplo normal:
    python main.py --city Chihuahua --lat 28.6353 --lon -106.0889 --airport CUU --hours 24
Ejemplo con MPI:
    mpiexec -n 4 python main.py --city Chihuahua --lat 28.6353 --lon -106.0889 --airport CUU --hours 24
"""
import argparse
from pathlib import Path

from src.config import Settings
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Análisis paralelo de tráfico aéreo y ambiente")
    parser.add_argument("--city", default="Chihuahua")
    parser.add_argument("--lat", type=float, default=28.6353)
    parser.add_argument("--lon", type=float, default=-106.0889)
    parser.add_argument("--airport", default="CUU")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output", default="salidas")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings(city=args.city, lat=args.lat, lon=args.lon, airport=args.airport, hours=args.hours, output_dir=Path(args.output))
    run_pipeline(settings, save=True)


if __name__ == "__main__":
    main()
