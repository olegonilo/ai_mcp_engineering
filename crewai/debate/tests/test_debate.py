import sys
import unittest
from unittest.mock import patch


class TestDebateCrew(unittest.TestCase):
    def test_crew_instantiates(self):
        from debate.crew import Debate
        crew = Debate().crew()
        self.assertIsNotNone(crew)

    def test_crew_has_two_agents(self):
        from debate.crew import Debate
        crew = Debate().crew()
        self.assertEqual(len(crew.agents), 2)

    def test_crew_has_three_tasks(self):
        from debate.crew import Debate
        crew = Debate().crew()
        self.assertEqual(len(crew.tasks), 3)

    def test_task_order_is_propose_oppose_decide(self):
        from debate.crew import Debate
        crew = Debate().crew()
        names = [t.name for t in crew.tasks]
        self.assertEqual(names, ["propose", "oppose", "decide"])

    def test_decide_task_has_context(self):
        from debate.crew import Debate
        decide = Debate().decide()
        self.assertIsNotNone(decide.context)
        self.assertEqual(len(decide.context), 2)

    def test_agent_roles(self):
        from debate.crew import Debate
        crew = Debate().crew()
        roles = {a.role.strip() for a in crew.agents}
        self.assertIn("A compelling debater", roles)
        self.assertIn("Impartial debate judge", roles)

    def test_tasks_have_output_files(self):
        from debate.crew import Debate
        debate = Debate()
        self.assertIn("propose.md", debate.propose().output_file)
        self.assertIn("oppose.md", debate.oppose().output_file)
        self.assertIn("decide.md", debate.decide().output_file)

    def test_crew_process_is_sequential(self):
        from crewai import Process
        from debate.crew import Debate
        crew = Debate().crew()
        self.assertEqual(crew.process, Process.sequential)


class TestEntryPoints(unittest.TestCase):
    def test_all_entry_points_exist(self):
        import debate.main as m
        for fn in ("run", "run_with_trigger", "train", "replay", "test"):
            self.assertTrue(callable(getattr(m, fn, None)), f"Missing: {fn}")

    def test_run_with_trigger_uses_argv(self):
        from debate import main
        captured = {}

        def fake_kickoff(inputs):
            captured["inputs"] = inputs

        with patch("debate.main.Debate") as MockDebate:
            MockDebate.return_value.crew.return_value.kickoff = fake_kickoff
            with patch.object(sys, "argv", ["debate", "AI should be open-source"]):
                main.run_with_trigger()

        self.assertEqual(captured["inputs"]["motion"], "AI should be open-source")

    def test_run_with_trigger_uses_default_when_no_argv(self):
        from debate import main
        captured = {}

        def fake_kickoff(inputs):
            captured["inputs"] = inputs

        with patch("debate.main.Debate") as MockDebate:
            MockDebate.return_value.crew.return_value.kickoff = fake_kickoff
            with patch.object(sys, "argv", ["debate"]):
                main.run_with_trigger()

        self.assertIn("LLMs", captured["inputs"]["motion"])


if __name__ == "__main__":
    unittest.main()
