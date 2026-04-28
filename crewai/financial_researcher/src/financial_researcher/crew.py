from pathlib import Path
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

_OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@CrewBase
class ResearchCrew:
    """Financial research crew for company analysis and reporting"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        # agents_config is replaced with a dict by CrewBase at runtime
        return Agent(
            config=self.agents_config["researcher"],  # type: ignore[index]
            tools=[SerperDevTool()],
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["analyst"],  # type: ignore[index]
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["analysis_task"],  # type: ignore[index]
            output_file=str(_OUTPUT_DIR / "report.md"),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
