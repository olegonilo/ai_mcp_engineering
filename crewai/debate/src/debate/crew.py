from pathlib import Path
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

_OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


@CrewBase
class Debate:
    """Debate crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def debater(self) -> Agent:
        return Agent(
            config=self.agents_config["debater"],  # type: ignore[index]
            max_iter=5,
        )

    @agent
    def judge(self) -> Agent:
        return Agent(
            config=self.agents_config["judge"],  # type: ignore[index]
            max_iter=3,
        )

    @task
    def propose(self) -> Task:
        return Task(
            config=self.tasks_config["propose"],  # type: ignore[index]
        )

    @task
    def oppose(self) -> Task:
        return Task(
            config=self.tasks_config["oppose"],  # type: ignore[index]
        )

    @task
    def decide(self) -> Task:
        return Task(
            config=self.tasks_config["decide"],  # type: ignore[index]
            context=[self.propose(), self.oppose()],
        )

    @crew
    def crew(self) -> Crew:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
