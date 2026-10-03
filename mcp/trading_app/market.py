import logging
import random
from datetime import datetime, timezone
from functools import lru_cache

from polygon import RESTClient

from config import MARKET_SIMULATION, POLYGON_API_KEY, is_paid_polygon, is_realtime_polygon
from database import read_market, write_market

logger = logging.getLogger(__name__)


class PriceUnavailableError(RuntimeError):
    pass


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


@lru_cache(maxsize=1)
def _client() -> RESTClient:
    if not POLYGON_API_KEY:
        raise PriceUnavailableError("POLYGON_API_KEY is not set")
    return RESTClient(POLYGON_API_KEY)


def is_market_open() -> bool:
    return _client().get_market_status().market == "open"


def get_all_share_prices_polygon_eod() -> dict[str, float]:
    """With much thanks to student Reema R. for fixing the timezone issue with this!"""
    client = _client()
    probe = client.get_previous_close_agg("SPY")[0]
    last_close = datetime.fromtimestamp(probe.timestamp / 1000, tz=timezone.utc).date()
    results = client.get_grouped_daily_aggs(last_close, adjusted=True, include_otc=False)
    return {result.ticker: result.close for result in results}


@lru_cache(maxsize=2)
def get_market_for_prior_date(today: str) -> dict[str, float]:
    market_data = read_market(today)
    if not market_data:
        market_data = get_all_share_prices_polygon_eod()
        write_market(today, market_data)
    return market_data


def get_share_price_polygon_eod(symbol: str) -> float:
    today = datetime.now().date().strftime("%Y-%m-%d")
    return get_market_for_prior_date(today).get(symbol, 0.0)


def get_share_price_polygon_snapshot(symbol: str) -> float:
    result = _client().get_snapshot_ticker("stocks", symbol)
    minute_close = result.min.close if result.min else None
    prev_close = result.prev_day.close if result.prev_day else None
    return minute_close or prev_close or 0.0


def get_share_price(symbol: str) -> float:
    """Return the latest price, or raise PriceUnavailableError. Never returns 0 or a fake price
    unless MARKET_SIMULATION is explicitly enabled."""
    symbol = normalize_symbol(symbol)
    if MARKET_SIMULATION and not POLYGON_API_KEY:
        return float(random.randint(1, 100))
    try:
        if is_paid_polygon or is_realtime_polygon:
            price = get_share_price_polygon_snapshot(symbol)
        else:
            price = get_share_price_polygon_eod(symbol)
    except PriceUnavailableError:
        raise
    except Exception as e:
        logger.exception("Polygon price lookup failed for %s", symbol)
        raise PriceUnavailableError(f"Price lookup failed for {symbol}: {e}") from e
    if not price or price <= 0:
        raise PriceUnavailableError(f"Unrecognized symbol {symbol}")
    return float(price)
