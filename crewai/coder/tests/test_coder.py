import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestCoderCrew(unittest.TestCase):
    def _crew(self):
        from coder.crew import Coder
        return Coder().crew()

    def test_crew_instantiates(self):
        self.assertIsNotNone(self._crew())

    def test_crew_has_one_agent(self):
        self.assertEqual(len(self._crew().agents), 1)

    def test_crew_has_one_task(self):
        self.assertEqual(len(self._crew().tasks), 1)

    def test_crew_process_is_sequential(self):
        from crewai import Process
        self.assertEqual(self._crew().process, Process.sequential)

    def test_agent_code_execution_mode_is_unsafe(self):
        # "unsafe" avoids a Docker dependency in production
        from coder.crew import Coder
        self.assertEqual(Coder().coder().code_execution_mode, "unsafe")

    def test_agent_max_iter(self):
        from coder.crew import Coder
        self.assertEqual(Coder().coder().max_iter, 5)

    def test_task_has_output_file(self):
        from coder.crew import Coder
        self.assertIn("code_and_output.txt", Coder().coding_task().output_file)

    def test_crew_loads_knowledge_when_file_exists(self):
        from coder.crew import Coder
        from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
        crew = Coder().crew()
        if Path("knowledge/user_preference.txt").exists() or \
                (Path(__file__).parent.parent / "knowledge" / "user_preference.txt").exists():
            self.assertTrue(
                any(isinstance(s, StringKnowledgeSource) for s in (crew.knowledge_sources or [])),
                "Expected StringKnowledgeSource when knowledge file exists",
            )

    def test_crew_skips_knowledge_when_file_absent(self):
        from coder.crew import _KNOWLEDGE_FILE, Coder
        with patch.object(type(_KNOWLEDGE_FILE), "exists", return_value=False):
            crew = Coder().crew()
        self.assertEqual(crew.knowledge_sources or [], [])


class TestEntryPoints(unittest.TestCase):
    def test_all_entry_points_exist(self):
        import coder.main as m
        for fn in ("run", "run_with_trigger", "train", "replay", "test"):
            self.assertTrue(callable(getattr(m, fn, None)), f"Missing: {fn}")

    def test_run_with_trigger_uses_argv(self):
        from coder import main

        mock_result = MagicMock()
        mock_result.raw = "test output"
        captured = {}

        def fake_kickoff(inputs):
            captured["inputs"] = inputs
            return mock_result

        with patch("coder.main.Coder") as MockCrew, \
             patch("coder.main._require_api_keys"):
            MockCrew.return_value.crew.return_value.kickoff = fake_kickoff
            with patch.object(sys, "argv", ["coder", "Write hello world"]):
                main.run_with_trigger()

        self.assertEqual(captured["inputs"]["assignment"], "Write hello world")

    def test_run_with_trigger_uses_default_when_no_argv(self):
        from coder import main

        mock_result = MagicMock()
        mock_result.raw = "test output"
        captured = {}

        def fake_kickoff(inputs):
            captured["inputs"] = inputs
            return mock_result

        with patch("coder.main.Coder") as MockCrew, \
             patch("coder.main._require_api_keys"):
            MockCrew.return_value.crew.return_value.kickoff = fake_kickoff
            with patch.object(sys, "argv", ["coder"]):
                main.run_with_trigger()

        self.assertIn("1 - 1/3", captured["inputs"]["assignment"])

    def test_require_api_keys_raises_when_missing(self):
        from coder.main import _require_api_keys
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(EnvironmentError) as ctx:
                _require_api_keys()
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_require_api_keys_passes_when_set(self):
        from coder.main import _require_api_keys
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            _require_api_keys()  # should not raise

    def test_train_exits_when_too_few_args(self):
        from coder import main
        with patch("coder.main._require_api_keys"), \
             patch.object(sys, "argv", ["train"]):
            with self.assertRaises(SystemExit) as ctx:
                main.train()
        self.assertIn("Usage", str(ctx.exception))

    def test_train_exits_on_bad_int(self):
        from coder import main
        with patch("coder.main._require_api_keys"), \
             patch.object(sys, "argv", ["train", "notanint", "output.json"]):
            with self.assertRaises(SystemExit) as ctx:
                main.train()
        self.assertIn("integer", str(ctx.exception))

    def test_replay_exits_when_too_few_args(self):
        from coder import main
        with patch("coder.main._require_api_keys"), \
             patch.object(sys, "argv", ["replay"]):
            with self.assertRaises(SystemExit) as ctx:
                main.replay()
        self.assertIn("Usage", str(ctx.exception))

    def test_test_exits_on_bad_int(self):
        from coder import main
        with patch("coder.main._require_api_keys"), \
             patch.object(sys, "argv", ["test", "notanint", "gpt-4o-mini"]):
            with self.assertRaises(SystemExit) as ctx:
                main.test()
        self.assertIn("integer", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
