#!/usr/bin/env python
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from coder.crew import Coder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_ASSIGNMENT = (
    "Write a python program to calculate the first 10,000 terms "
    "of this series, multiplying the total by 4: 1 - 1/3 + 1/5 - 1/7 + ..."
)
_REQUIRED_KEYS = ("OPENAI_API_KEY",)


def _require_api_keys() -> None:
    missing = [k for k in _REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"{', '.join(missing)} not set. Add them to a .env file or export before running."
        )


def run(assignment: str = _DEFAULT_ASSIGNMENT) -> None:
    _require_api_keys()
    result = Coder().crew().kickoff(inputs={"assignment": assignment})
    print("\n=== RESULT ===\n")
    print(result.raw)
    logger.info("Output saved to output/code_and_output.txt")


def run_with_trigger() -> None:
    assignment = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_ASSIGNMENT
    run(assignment)


def train() -> None:
    _require_api_keys()
    if len(sys.argv) < 3:
        raise SystemExit("Usage: train <n_iterations> <filename>")
    try:
        n_iterations = int(sys.argv[1])
    except ValueError:
        raise SystemExit(f"Error: n_iterations must be an integer, got '{sys.argv[1]}'")
    Coder().crew().train(
        n_iterations=n_iterations,
        filename=sys.argv[2],
        inputs={"assignment": _DEFAULT_ASSIGNMENT},
    )


def replay() -> None:
    _require_api_keys()
    if len(sys.argv) < 2:
        raise SystemExit("Usage: replay <task_id>")
    Coder().crew().replay(task_id=sys.argv[1])


def test() -> None:
    _require_api_keys()
    if len(sys.argv) < 3:
        raise SystemExit("Usage: test <n_iterations> <eval_llm>")
    try:
        n_iterations = int(sys.argv[1])
    except ValueError:
        raise SystemExit(f"Error: n_iterations must be an integer, got '{sys.argv[1]}'")
    Coder().crew().test(
        n_iterations=n_iterations,
        eval_llm=sys.argv[2],
        inputs={"assignment": _DEFAULT_ASSIGNMENT},
    )


if __name__ == "__main__":
    run()
