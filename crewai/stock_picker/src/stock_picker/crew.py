from pathlib import Path
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.memory import Memory
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field

from .tools.push_tool import PushNotificationTool

_OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
_MEMORY_DIR = Path(__file__).parent.parent.parent / "memory"

_EMBEDDER_CONFIG = {
    "provider": "openai",
    "config": {"model": "text-embedding-3-small"},
}


class TrendingCompany(BaseModel):
    name: str = Field(description="Company name")
    ticker: str = Field(description="Stock ticker symbol")
    reason: str = Field(description="Reason this company is trending in the news")


class TrendingCompanyList(BaseModel):
    companies: List[TrendingCompany] = Field(description="List of companies trending in the news")


class TrendingCompanyResearch(BaseModel):
    name: str = Field(description="Company name")
    market_position: str = Field(description="Current market position and competitive analysis")
    future_outlook: str = Field(description="Future outlook and growth prospects")
    investment_potential: str = Field(description="Investment potential and suitability for investment")


class TrendingCompanyResearchList(BaseModel):
    research_list: List[TrendingCompanyResearch] = Field(
        description="Comprehensive research on all trending companies"
    )


@CrewBase
class StockPicker:
    """StockPicker crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def trending_company_finder(self) -> Agent:
        return Agent(
            config=self.agents_config["trending_company_finder"],  # type: ignore[index]
            tools=[SerperDevTool()],
            memory=True,
        )

    @agent
    def financial_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["financial_researcher"],  # type: ignore[index]
            tools=[SerperDevTool()],
        )

    @agent
    def stock_picker(self) -> Agent:
        return Agent(
            config=self.agents_config["stock_picker"],  # type: ignore[index]
            tools=[PushNotificationTool()],
            memory=True,
        )

    @task
    def find_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config["find_trending_companies"],  # type: ignore[index]
            output_pydantic=TrendingCompanyList,
        )

    @task
    def research_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config["research_trending_companies"],  # type: ignore[index]
            output_pydantic=TrendingCompanyResearchList,
        )

    @task
    def pick_best_company(self) -> Task:
        return Task(
            config=self.tasks_config["pick_best_company"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the StockPicker crew"""
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        manager = Agent(
            config=self.agents_config["manager"],  # type: ignore[index]
            allow_delegation=True,
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            verbose=True,
            manager_agent=manager,
            memory=Memory(
                storage=str(_MEMORY_DIR),
                embedder=_EMBEDDER_CONFIG,
            ),
        )
