from pathlib import Path
from typing import List

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.project import CrewBase, agent, crew, task

from crewai import Agent, Crew, Process, Task

_OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
_KNOWLEDGE_FILE = Path(__file__).parent.parent.parent / "knowledge" / "user_preference.txt"


@CrewBase
class Coder:
    """Coder crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def coder(self) -> Agent:
        return Agent(
            config=self.agents_config["coder"],  # type: ignore[index]
            verbose=True,
            allow_code_execution=True,
            code_execution_mode="safe",
            max_retry_limit=3,
            max_iter=5,
        )

    @task
    def coding_task(self) -> Task:
        return Task(
            config=self.tasks_config["coding_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Coder crew"""
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        knowledge_sources = []
        if _KNOWLEDGE_FILE.exists():
            knowledge_sources = [StringKnowledgeSource(content=_KNOWLEDGE_FILE.read_text())]

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            knowledge_sources=knowledge_sources,
        )
