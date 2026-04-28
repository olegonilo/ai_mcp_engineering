#!/usr/bin/env python
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="crewai")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from financial_researcher.crew import ResearchCrew  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_COMPANY = "Apple"
_REQUIRED_KEYS = ("OPENAI_API_KEY", "SERPER_API_KEY")


def _require_api_keys() -> None:
    missing = [k for k in _REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"{', '.join(missing)} not set. Add them to a .env file or export before running."
        )


def run(company: str = _DEFAULT_COMPANY) -> None:
    _require_api_keys()
    result = ResearchCrew().crew().kickoff(inputs={"company": company})
    print("\n=== FINAL REPORT ===\n")
    print(result.raw)
    logger.info("Report saved to output/report.md")


def run_with_trigger() -> None:
    company = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_COMPANY
    run(company)


def train() -> None:
    _require_api_keys()
    if len(sys.argv) < 3:
        raise SystemExit("Usage: train <n_iterations> <filename>")
    ResearchCrew().crew().train(
        n_iterations=int(sys.argv[1]),
        filename=sys.argv[2],
        inputs={"company": _DEFAULT_COMPANY},
    )


def replay() -> None:
    _require_api_keys()
    if len(sys.argv) < 2:
        raise SystemExit("Usage: replay <task_id>")
    ResearchCrew().crew().replay(task_id=sys.argv[1])


def test() -> None:
    _require_api_keys()
    if len(sys.argv) < 3:
        raise SystemExit("Usage: test <n_iterations> <openai_model_name>")
    ResearchCrew().crew().test(
        n_iterations=int(sys.argv[1]),
        openai_model_name=sys.argv[2],
        inputs={"company": _DEFAULT_COMPANY},
    )


if __name__ == "__main__":
    run()
