from pydantic import BaseModel, Field
from agents import Agent

INSTRUCTIONS = (
    "You are a research assistant. Given a research query, generate exactly 3 concise clarifying questions "
    "that help focus the research on what matters most to the user. "
    "Ask about scope, timeframe, target audience, depth of coverage, or specific angles they care about."
)


class ClarifyingQuestions(BaseModel):
    questions: list[str] = Field(description="Exactly 3 clarifying questions to refine the research scope.")


clarifier_agent = Agent(
    name="ClarifierAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ClarifyingQuestions,
)
