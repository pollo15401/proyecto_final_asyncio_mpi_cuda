from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "salidas"
DATA_DIR = BASE_DIR / "data"

@dataclass
class Settings:
    city: str = os.getenv("DEFAULT_CITY", "Chihuahua")
    lat: float = float(os.getenv("DEFAULT_LAT", "28.6353"))
    lon: float = float(os.getenv("DEFAULT_LON", "-106.0889"))
    airport: str = os.getenv("DEFAULT_AIRPORT", "CUU")
    hours: int = 24
    output_dir: Path = OUTPUT_DIR
    airlabs_api_key: str = os.getenv("AIRLABS_API_KEY", "")
    openweather_api_key: str = os.getenv("OPENWEATHER_API_KEY", "")
    openaq_api_key: str = os.getenv("OPENAQ_API_KEY", "")
    thingspeak_channel_id: str = os.getenv("THINGSPEAK_CHANNEL_ID", "12397")
    thingspeak_read_api_key: str = os.getenv("THINGSPEAK_READ_API_KEY", "")

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "graficas").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "csv").mkdir(parents=True, exist_ok=True)
