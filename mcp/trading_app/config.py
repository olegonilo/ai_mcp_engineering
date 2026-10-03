"""Single source of configuration for the trading app.

Every module imports settings from here instead of calling load_dotenv() itself.
Paths are absolute so the app works regardless of the current working directory.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Real environment variables (containers, CI, secret managers) must win over .env.
load_dotenv(override=False)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


DB_PATH = Path(os.getenv("TRADING_DB_PATH", BASE_DIR / "accounts.db"))
MEMORY_DIR = BASE_DIR / "memory"

INITIAL_BALANCE = 10_000.0
SPREAD = 0.002
MAX_TIME_SERIES_POINTS = 5_000
REPORT_TRANSACTIONS = 20

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
POLYGON_PLAN = (os.getenv("POLYGON_PLAN") or "").strip().lower()
is_paid_polygon = POLYGON_PLAN == "paid"
is_realtime_polygon = POLYGON_PLAN == "realtime"
# Explicit opt-in: random prices are only acceptable for local demos, never in production.
MARKET_SIMULATION = env_flag("MARKET_SIMULATION")

RUN_EVERY_N_MINUTES = int(os.getenv("RUN_EVERY_N_MINUTES", "60"))
RUN_EVEN_WHEN_MARKET_IS_CLOSED = env_flag("RUN_EVEN_WHEN_MARKET_IS_CLOSED")
USE_MANY_MODELS = env_flag("USE_MANY_MODELS")
MAX_TURNS = int(os.getenv("MAX_TURNS", "30"))
TRADER_RUN_TIMEOUT_SECONDS = int(os.getenv("TRADER_RUN_TIMEOUT_SECONDS", "900"))
MCP_SESSION_TIMEOUT_SECONDS = int(os.getenv("MCP_SESSION_TIMEOUT_SECONDS", "120"))
