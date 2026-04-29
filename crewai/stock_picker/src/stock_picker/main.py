#!/usr/bin/env python
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="crewai")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from stock_picker.crew import StockPicker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_SECTOR = "Technology"
_REQUIRED_KEYS = ("OPENAI_API_KEY", "SERPER_API_KEY", "PUSHOVER_USER", "PUSHOVER_TOKEN")


def _require_api_keys() -> None:
    missing = [k for k in _REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"{', '.join(missing)} not set. Add them to a .env file or export before running."
        )


def run(sector: str = _DEFAULT_SECTOR) -> None:
    _require_api_keys()
    result = StockPicker().crew().kickoff(
        inputs={"sector": sector, "current_date": str(datetime.now().date())}
    )
    print("\n=== FINAL DECISION ===\n")
    print(result.raw)
    logger.info("Decision saved to output/decision.md (relative to project root)")


def run_with_trigger() -> None:
    sector = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_SECTOR
    run(sector)


def train() -> None:
    _require_api_keys()
    if len(sys.argv) < 3:
        raise SystemExit("Usage: train <n_iterations> <filename>")
    try:
        n_iterations = int(sys.argv[1])
    except ValueError:
        raise SystemExit(f"Error: n_iterations must be an integer, got '{sys.argv[1]}'")
    StockPicker().crew().train(
        n_iterations=n_iterations,
        filename=sys.argv[2],
        inputs={"sector": _DEFAULT_SECTOR},
    )


def replay() -> None:
    _require_api_keys()
    if len(sys.argv) < 2:
        raise SystemExit("Usage: replay <task_id>")
    StockPicker().crew().replay(task_id=sys.argv[1])


def test() -> None:
    _require_api_keys()
    if len(sys.argv) < 3:
        raise SystemExit("Usage: test <n_iterations> <eval_llm>")
    try:
        n_iterations = int(sys.argv[1])
    except ValueError:
        raise SystemExit(f"Error: n_iterations must be an integer, got '{sys.argv[1]}'")
    StockPicker().crew().test(
        n_iterations=n_iterations,
        eval_llm=sys.argv[2],
        inputs={"sector": _DEFAULT_SECTOR},
    )


if __name__ == "__main__":
    run()
