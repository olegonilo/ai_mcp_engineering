"""Tests for engineering_team crew — structure and utility functions."""
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from engineering_team.main import (
    CLASS_NAME,
    MODULE_NAME,
    OUTPUT_DIR,
    REQUIREMENTS,
    _strip_fences,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write(tmp: Path, content: str, name: str = "test.py") -> Path:
    p = tmp / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# _strip_fences — happy paths
# ---------------------------------------------------------------------------

def test_strip_fences_removes_python_fence(tmp_path):
    p = _write(tmp_path, "```python\nprint('hello')\n```")
    _strip_fences(p)
    assert p.read_text().strip() == "print('hello')"


def test_strip_fences_removes_plain_fence(tmp_path):
    p = _write(tmp_path, "```\nx = 1\n```")
    _strip_fences(p)
    assert p.read_text().strip() == "x = 1"


def test_strip_fences_leaves_clean_code_unchanged(tmp_path):
    code = "x = 1\nprint(x)\n"
    p = _write(tmp_path, code)
    _strip_fences(p)
    assert p.read_text() == code


def test_strip_fences_trailing_newline_always_present(tmp_path):
    p = _write(tmp_path, "x = 1")
    _strip_fences(p)
    assert p.read_text().endswith("\n")


def test_strip_fences_multiline_code_preserved(tmp_path):
    code = "def foo():\n    return 42\n\nfoo()\n"
    p = _write(tmp_path, f"```python\n{code}```")
    _strip_fences(p)
    assert "def foo():" in p.read_text()
    assert "return 42" in p.read_text()


# ---------------------------------------------------------------------------
# _strip_fences — edge cases
# ---------------------------------------------------------------------------

def test_strip_fences_handles_empty_file(tmp_path):
    p = _write(tmp_path, "")
    _strip_fences(p)
    assert p.read_text() == "\n"


def test_strip_fences_fence_with_trailing_spaces(tmp_path):
    p = _write(tmp_path, "```python  \nx = 1\n```")
    _strip_fences(p)
    assert p.read_text().strip() == "x = 1"


def test_strip_fences_does_not_strip_internal_backticks(tmp_path):
    code = 'x = "`not a fence`"\n'
    p = _write(tmp_path, code)
    _strip_fences(p)
    assert "`not a fence`" in p.read_text()


def test_strip_fences_idempotent(tmp_path):
    p = _write(tmp_path, "x = 1\n")
    _strip_fences(p)
    first = p.read_text()
    _strip_fences(p)
    assert p.read_text() == first


def test_strip_fences_opening_fence_only(tmp_path):
    p = _write(tmp_path, "```python\nx = 1\n")
    _strip_fences(p)
    assert "```" not in p.read_text()


def test_strip_fences_closing_fence_only(tmp_path):
    p = _write(tmp_path, "x = 1\n```")
    _strip_fences(p)
    assert "```" not in p.read_text()


def test_strip_fences_whitespace_only_file(tmp_path):
    p = _write(tmp_path, "   \n\n   ")
    _strip_fences(p)
    assert p.read_text() == "\n"


# ---------------------------------------------------------------------------
# Constants — correctness and completeness
# ---------------------------------------------------------------------------

def test_module_name():
    assert MODULE_NAME == "tictactoe.py"


def test_class_name():
    assert CLASS_NAME == "TicTacToe"


def test_requirements_mentions_tictactoe():
    assert "Tic-Tac-Toe" in REQUIREMENTS


def test_requirements_mentions_gradio():
    assert "Gradio" in REQUIREMENTS


def test_requirements_describes_win_detection():
    assert "win" in REQUIREMENTS.lower()


def test_requirements_describes_draw_detection():
    assert "draw" in REQUIREMENTS.lower()


def test_requirements_describes_reset():
    assert "reset" in REQUIREMENTS.lower() or "new game" in REQUIREMENTS.lower()


def test_requirements_describes_board_state():
    assert "board" in REQUIREMENTS.lower()


def test_output_dir_is_absolute():
    assert OUTPUT_DIR.is_absolute()


def test_output_dir_named_output():
    assert OUTPUT_DIR.name == "output"


# ---------------------------------------------------------------------------
# run() — mocked to avoid real LLM calls
# ---------------------------------------------------------------------------

def _make_run_mocks():
    """Return (mock_team_cls, mock_crew_obj, mock_dir) for patching run()."""
    mock_crew_obj = MagicMock()
    mock_team_cls = MagicMock()
    mock_team_cls.return_value.crew.return_value = mock_crew_obj
    mock_dir = MagicMock(spec=Path)
    mock_dir.glob.return_value = []
    return mock_team_cls, mock_crew_obj, mock_dir


def test_run_calls_kickoff_with_correct_inputs():
    mock_team_cls, mock_crew_obj, mock_dir = _make_run_mocks()
    with patch("engineering_team.main.EngineeringTeam", mock_team_cls), \
         patch("engineering_team.main.OUTPUT_DIR", mock_dir):
        from engineering_team.main import run
        run()
    mock_crew_obj.kickoff.assert_called_once_with(
        inputs={
            "requirements": REQUIREMENTS,
            "module_name": MODULE_NAME,
            "class_name": CLASS_NAME,
        }
    )


def test_run_creates_output_dir():
    mock_team_cls, _, mock_dir = _make_run_mocks()
    with patch("engineering_team.main.EngineeringTeam", mock_team_cls), \
         patch("engineering_team.main.OUTPUT_DIR", mock_dir):
        from engineering_team.main import run
        run()
    mock_dir.mkdir.assert_called_once_with(exist_ok=True)


def test_run_calls_strip_fences_for_each_py_file(tmp_path):
    mock_team_cls, _, mock_dir = _make_run_mocks()
    py_files = [tmp_path / "a.py", tmp_path / "b.py"]
    for f in py_files:
        f.write_text("x = 1\n")
    mock_dir.glob.return_value = py_files

    with patch("engineering_team.main.EngineeringTeam", mock_team_cls), \
         patch("engineering_team.main.OUTPUT_DIR", mock_dir), \
         patch("engineering_team.main._strip_fences") as mock_strip:
        from engineering_team.main import run
        run()

    assert mock_strip.call_count == 2
    mock_strip.assert_any_call(py_files[0])
    mock_strip.assert_any_call(py_files[1])


def test_run_strip_fences_glob_uses_py_pattern():
    mock_team_cls, _, mock_dir = _make_run_mocks()
    with patch("engineering_team.main.EngineeringTeam", mock_team_cls), \
         patch("engineering_team.main.OUTPUT_DIR", mock_dir):
        from engineering_team.main import run
        run()
    mock_dir.glob.assert_called_once_with("*.py")


def test_run_reraises_kickoff_exception():
    mock_team_cls, mock_crew_obj, mock_dir = _make_run_mocks()
    mock_crew_obj.kickoff.side_effect = RuntimeError("LLM timeout")
    with patch("engineering_team.main.EngineeringTeam", mock_team_cls), \
         patch("engineering_team.main.OUTPUT_DIR", mock_dir):
        from engineering_team.main import run
        with pytest.raises(RuntimeError, match="LLM timeout"):
            run()


def test_run_does_not_strip_fences_on_kickoff_failure():
    mock_team_cls, mock_crew_obj, mock_dir = _make_run_mocks()
    mock_crew_obj.kickoff.side_effect = RuntimeError("boom")
    with patch("engineering_team.main.EngineeringTeam", mock_team_cls), \
         patch("engineering_team.main.OUTPUT_DIR", mock_dir), \
         patch("engineering_team.main._strip_fences") as mock_strip:
        from engineering_team.main import run
        with pytest.raises(RuntimeError):
            run()
    mock_strip.assert_not_called()


def test_run_no_strip_when_no_py_files():
    mock_team_cls, _, mock_dir = _make_run_mocks()
    mock_dir.glob.return_value = []
    with patch("engineering_team.main.EngineeringTeam", mock_team_cls), \
         patch("engineering_team.main.OUTPUT_DIR", mock_dir), \
         patch("engineering_team.main._strip_fences") as mock_strip:
        from engineering_team.main import run
        run()
    mock_strip.assert_not_called()


# ---------------------------------------------------------------------------
# Crew structure — agents
# ---------------------------------------------------------------------------

def test_crew_has_four_agents():
    from engineering_team.crew import EngineeringTeam
    assert len(EngineeringTeam().crew().agents) == 4


def test_crew_has_four_tasks():
    from engineering_team.crew import EngineeringTeam
    assert len(EngineeringTeam().crew().tasks) == 4


def test_all_agent_roles_present():
    from engineering_team.crew import EngineeringTeam
    roles = " ".join(a.role for a in EngineeringTeam().crew().agents).lower()
    assert "engineering lead" in roles or "lead" in roles
    assert "engineer" in roles


def test_backend_engineer_max_retry_limit():
    from engineering_team.crew import EngineeringTeam
    team = EngineeringTeam()
    agent = team.backend_engineer()
    assert agent.max_retry_limit == 3


def test_test_engineer_max_retry_limit():
    from engineering_team.crew import EngineeringTeam
    team = EngineeringTeam()
    agent = team.test_engineer()
    assert agent.max_retry_limit == 3


def test_all_agents_use_expected_llm():
    from engineering_team.crew import EngineeringTeam
    team = EngineeringTeam()
    for agent in team.crew().agents:
        llm = str(agent.llm).lower()
        assert "gpt-4o-mini" in llm or "openai" in llm, f"Unexpected LLM for agent '{agent.role}': {agent.llm}"


# ---------------------------------------------------------------------------
# Crew structure — tasks and dependencies
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def crew_tasks():
    from engineering_team.crew import EngineeringTeam
    crew = EngineeringTeam().crew()
    return {t.name: t for t in crew.tasks}


def test_design_task_exists(crew_tasks):
    assert "design_task" in crew_tasks


def test_code_task_exists(crew_tasks):
    assert "code_task" in crew_tasks


def test_frontend_task_exists(crew_tasks):
    assert "frontend_task" in crew_tasks


def test_test_task_exists(crew_tasks):
    assert "test_task" in crew_tasks


def test_code_task_depends_on_design_task(crew_tasks):
    ctx = crew_tasks["code_task"].context or []
    assert crew_tasks["design_task"] in ctx


def test_frontend_task_depends_on_code_task(crew_tasks):
    ctx = crew_tasks["frontend_task"].context or []
    assert crew_tasks["code_task"] in ctx


def test_test_task_depends_on_code_task(crew_tasks):
    ctx = crew_tasks["test_task"].context or []
    assert crew_tasks["code_task"] in ctx


def test_design_task_has_no_context(crew_tasks):
    ctx = crew_tasks["design_task"].context
    # crewai uses a NOT_SPECIFIED sentinel when no context is declared;
    # treat any non-list value as "unset", and an empty list as also valid
    if isinstance(ctx, list):
        assert ctx == [], f"design_task should have no context dependencies, got {ctx}"


def test_crew_process_is_sequential():
    from crewai import Process
    from engineering_team.crew import EngineeringTeam
    assert EngineeringTeam().crew().process == Process.sequential


def test_task_output_files_cover_module_and_app(crew_tasks):
    output_files = [t.output_file for t in crew_tasks.values() if t.output_file]
    joined = " ".join(output_files)
    assert "app.py" in joined
    assert "tictactoe" in joined or "module_name" in joined
