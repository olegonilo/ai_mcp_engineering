#!/usr/bin/env python
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="crewai")

from dotenv import load_dotenv

load_dotenv()

from debate.crew import Debate  # noqa: E402

_DEFAULT_MOTION = "There needs to be strict laws to regulate LLMs"


def _require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to a .env file or export it before running."
        )


def run() -> None:
    _require_api_key()
    inputs = {"motion": _DEFAULT_MOTION}
    try:
        result = Debate().crew().kickoff(inputs=inputs)
        print("\n=== DEBATE RESULT ===\n")
        print(result)
    except Exception as e:
        raise RuntimeError(f"An error occurred while running the crew: {e}") from e


def run_with_trigger() -> None:
    """Run with a custom motion passed as the first CLI argument."""
    _require_api_key()
    motion = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_MOTION
    inputs = {"motion": motion}
    try:
        result = Debate().crew().kickoff(inputs=inputs)
        print("\n=== DEBATE RESULT ===\n")
        print(result)
    except Exception as e:
        raise RuntimeError(f"An error occurred while running the crew: {e}") from e


def train() -> None:
    _require_api_key()
    if len(sys.argv) < 3:
        raise SystemExit("Usage: train <n_iterations> <filename>")
    inputs = {"motion": _DEFAULT_MOTION}
    try:
        Debate().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs,
        )
    except Exception as e:
        raise RuntimeError(f"An error occurred while training the crew: {e}") from e


def replay() -> None:
    _require_api_key()
    if len(sys.argv) < 2:
        raise SystemExit("Usage: replay <task_id>")
    try:
        Debate().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise RuntimeError(f"An error occurred while replaying the crew: {e}") from e


def test() -> None:
    _require_api_key()
    if len(sys.argv) < 3:
        raise SystemExit("Usage: test <n_iterations> <openai_model_name>")
    inputs = {"motion": _DEFAULT_MOTION}
    try:
        Debate().crew().test(
            n_iterations=int(sys.argv[1]),
            openai_model_name=sys.argv[2],
            inputs=inputs,
        )
    except Exception as e:
        raise RuntimeError(f"An error occurred while testing the crew: {e}") from e
