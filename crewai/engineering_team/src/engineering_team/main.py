#!/usr/bin/env python
import logging
import re
import warnings
from pathlib import Path

from engineering_team.crew import EngineeringTeam

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output"

REQUIREMENTS = """
A Tic-Tac-Toe game for two players (X and O) played on a 3x3 board.
The class must support:
- Starting a new game / resetting the board.
- Making a move for the current player given a row and column (0-indexed).
- Validating that a move is legal (cell is empty, game is not over).
- Detecting a win (any row, column, or diagonal fully owned by one player).
- Detecting a draw (board full with no winner).
- Returning the current board state as a 3x3 list of strings (' ', 'X', 'O').
- Returning whose turn it is ('X' or 'O').
- Returning the game status: 'ongoing', 'X wins', 'O wins', or 'draw'.
The Gradio UI should let two human players click buttons (one per cell) to make moves,
display the board visually, show whose turn it is, announce the winner or draw,
and provide a Reset button to start a new game.
"""
MODULE_NAME = "tictactoe.py"
CLASS_NAME = "TicTacToe"


def _strip_fences(path: Path) -> None:
    """Strip markdown code fences that LLMs sometimes emit despite instructions."""
    text = path.read_text()
    text = re.sub(r"^```[^\n]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    path.write_text(text + "\n")


def run() -> None:
    """Run the engineering team crew to generate the tic-tac-toe game."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    inputs = {
        "requirements": REQUIREMENTS,
        "module_name": MODULE_NAME,
        "class_name": CLASS_NAME,
    }

    try:
        result = EngineeringTeam().crew().kickoff(inputs=inputs)
        log.info("Crew finished successfully. Result: %s", result)
    except Exception:
        log.exception("Crew execution failed")
        raise

    for py_file in OUTPUT_DIR.glob("*.py"):
        _strip_fences(py_file)
        log.info("Output written: %s", py_file)


if __name__ == "__main__":
    run()
