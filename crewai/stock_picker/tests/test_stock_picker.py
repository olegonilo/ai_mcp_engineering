import json
import sys
import unittest
from unittest.mock import MagicMock, patch


class TestStockPickerCrew(unittest.TestCase):
    def _crew(self):
        from stock_picker.crew import StockPicker
        return StockPicker().crew()

    def test_crew_instantiates(self):
        self.assertIsNotNone(self._crew())

    def test_crew_has_three_agents(self):
        self.assertEqual(len(self._crew().agents), 3)

    def test_crew_has_three_tasks(self):
        self.assertEqual(len(self._crew().tasks), 3)

    def test_task_order(self):
        names = [t.name for t in self._crew().tasks]
        self.assertEqual(names, [
            "find_trending_companies",
            "research_trending_companies",
            "pick_best_company",
        ])

    def test_crew_process_is_hierarchical(self):
        from crewai import Process
        self.assertEqual(self._crew().process, Process.hierarchical)

    def test_crew_has_memory(self):
        self.assertIsNotNone(self._crew().memory)

    def test_tasks_have_output_files(self):
        from stock_picker.crew import StockPicker
        sp = StockPicker()
        self.assertIn("trending_companies.json", sp.find_trending_companies().output_file)
        self.assertIn("research_report.json", sp.research_trending_companies().output_file)
        self.assertIn("decision.md", sp.pick_best_company().output_file)

    def test_agents_have_max_iter(self):
        from stock_picker.crew import StockPicker
        sp = StockPicker()
        self.assertEqual(sp.trending_company_finder().max_iter, 5)
        self.assertEqual(sp.financial_researcher().max_iter, 5)
        self.assertEqual(sp.stock_picker().max_iter, 3)

    def test_agents_do_not_have_individual_memory(self):
        # Agents must NOT set memory=True — they share crew-level memory
        from stock_picker.crew import StockPicker
        sp = StockPicker()
        self.assertIsNone(sp.trending_company_finder().memory)
        self.assertIsNone(sp.financial_researcher().memory)
        self.assertIsNone(sp.stock_picker().memory)

    def test_research_task_has_context(self):
        from stock_picker.crew import StockPicker
        task = StockPicker().research_trending_companies()
        self.assertIsNotNone(task.context)
        self.assertGreater(len(task.context), 0)

    def test_pick_task_has_context(self):
        from stock_picker.crew import StockPicker
        task = StockPicker().pick_best_company()
        self.assertIsNotNone(task.context)
        self.assertGreater(len(task.context), 0)


class TestEntryPoints(unittest.TestCase):
    def test_all_entry_points_exist(self):
        import stock_picker.main as m
        for fn in ("run", "run_with_trigger", "train", "replay", "test"):
            self.assertTrue(callable(getattr(m, fn, None)), f"Missing: {fn}")

    def test_run_with_trigger_uses_argv(self):
        from stock_picker import main
        captured = {}

        mock_result = MagicMock()
        mock_result.raw = "test decision"

        def fake_kickoff(inputs):
            captured["inputs"] = inputs
            return mock_result

        with patch("stock_picker.main.StockPicker") as MockCrew, \
             patch("stock_picker.main.build_memory"), \
             patch("stock_picker.main.seed_user_memory"), \
             patch("stock_picker.main._require_api_keys"):
            MockCrew.return_value.crew.return_value.kickoff = fake_kickoff
            with patch.object(sys, "argv", ["stock_picker", "Healthcare"]):
                main.run_with_trigger()

        self.assertEqual(captured["inputs"]["sector"], "Healthcare")

    def test_run_with_trigger_uses_default_when_no_argv(self):
        from stock_picker import main
        captured = {}

        mock_result = MagicMock()
        mock_result.raw = "test decision"

        def fake_kickoff(inputs):
            captured["inputs"] = inputs
            return mock_result

        with patch("stock_picker.main.StockPicker") as MockCrew, \
             patch("stock_picker.main.build_memory"), \
             patch("stock_picker.main.seed_user_memory"), \
             patch("stock_picker.main._require_api_keys"):
            MockCrew.return_value.crew.return_value.kickoff = fake_kickoff
            with patch.object(sys, "argv", ["stock_picker"]):
                main.run_with_trigger()

        self.assertEqual(captured["inputs"]["sector"], "Technology")


class TestPushNotificationTool(unittest.TestCase):
    def test_missing_credentials_returns_error(self):
        from stock_picker.tools.push_tool import PushNotificationTool
        tool = PushNotificationTool()
        with patch.dict("os.environ", {}, clear=True):
            result = json.loads(tool._run("test message"))
        self.assertEqual(result["notification"], "error")
        self.assertIn("PUSHOVER", result["reason"])

    def test_successful_notification(self):
        from stock_picker.tools.push_tool import PushNotificationTool
        tool = PushNotificationTool()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        with patch("stock_picker.tools.push_tool.requests.post", return_value=mock_response), \
             patch.dict("os.environ", {"PUSHOVER_USER": "u", "PUSHOVER_TOKEN": "t"}):
            result = json.loads(tool._run("buy AAPL"))
        self.assertEqual(result["notification"], "ok")

    def test_network_error_returns_error(self):
        import requests
        from stock_picker.tools.push_tool import PushNotificationTool
        tool = PushNotificationTool()
        with patch("stock_picker.tools.push_tool.requests.post",
                   side_effect=requests.RequestException("timeout")), \
             patch.dict("os.environ", {"PUSHOVER_USER": "u", "PUSHOVER_TOKEN": "t"}):
            result = json.loads(tool._run("buy AAPL"))
        self.assertEqual(result["notification"], "error")
        self.assertIn("timeout", result["reason"])


class TestMemoryModule(unittest.TestCase):
    def test_build_memory_returns_memory_instance(self):
        from crewai.memory import Memory
        from stock_picker.memory import build_memory
        self.assertIsInstance(build_memory(), Memory)

    def test_memory_scopes_return_memory_scope(self):
        from crewai.memory.memory_scope import MemoryScope
        from stock_picker.memory import build_memory, long_term, short_term, user
        mem = build_memory()
        self.assertIsInstance(short_term(mem), MemoryScope)
        self.assertIsInstance(long_term(mem), MemoryScope)
        self.assertIsInstance(user(mem), MemoryScope)

    def test_seed_user_memory_does_not_raise(self):
        from stock_picker.memory import build_memory, seed_user_memory
        mem = build_memory()
        with patch("stock_picker.memory.MemoryScope.remember_many", return_value=[]):
            seed_user_memory(mem)


if __name__ == "__main__":
    unittest.main()
