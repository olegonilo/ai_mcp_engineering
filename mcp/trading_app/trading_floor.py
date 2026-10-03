import asyncio
import logging
import time

from agents import add_trace_processor

from config import RUN_EVEN_WHEN_MARKET_IS_CLOSED, RUN_EVERY_N_MINUTES, USE_MANY_MODELS
from market import is_market_open
from traders import Trader
from tracers import LogTracer

logger = logging.getLogger(__name__)

names = ["Warren", "George", "Ray", "Cathie"]
lastnames = ["Patience", "Bold", "Systematic", "Crypto"]

if USE_MANY_MODELS:
    # Preview model ids get retired by providers; verify availability before each deployment.
    model_names = [
        "gpt-4.1-mini",
        "deepseek-chat",
        "gemini-2.5-flash-preview-04-17",
        "grok-3-mini-beta",
    ]
    short_model_names = ["GPT 4.1 Mini", "DeepSeek V3", "Gemini 2.5 Flash", "Grok 3 Mini"]
else:
    model_names = ["gpt-4o-mini"] * 4
    short_model_names = ["GPT 4o mini"] * 4


def create_traders() -> list[Trader]:
    return [Trader(name, lastname, model_name) for name, lastname, model_name in zip(names, lastnames, model_names, strict=True)]


def should_run() -> bool:
    if RUN_EVEN_WHEN_MARKET_IS_CLOSED:
        return True
    try:
        return is_market_open()
    except Exception:
        # A market-status outage must not kill the scheduler loop.
        logger.exception("Could not determine market status; skipping run")
        return False


async def run_every_n_minutes():
    add_trace_processor(LogTracer(names))
    traders = create_traders()
    interval = RUN_EVERY_N_MINUTES * 60
    while True:
        started = time.monotonic()
        if should_run():
            # Trader.run handles its own errors and timeout, so one trader cannot stop the others.
            await asyncio.gather(*(trader.run() for trader in traders))
        else:
            logger.info("Market is closed, skipping run")
        # Keep a fixed cadence instead of drifting by the duration of each run.
        await asyncio.sleep(max(0.0, interval - (time.monotonic() - started)))


if __name__ == "__main__":
    logger.info("Starting scheduler to run every %s minutes", RUN_EVERY_N_MINUTES)
    asyncio.run(run_every_n_minutes())
