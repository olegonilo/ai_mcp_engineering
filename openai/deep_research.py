import asyncio
import os
import sys

import sendgrid
from agents import Agent, Runner, WebSearchTool, function_tool, trace
from agents.model_settings import ModelSettings
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sendgrid.helpers.mail import Content, Email, Mail, To

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    sys.exit("OPENAI_API_KEY is required but not set")

FROM_EMAIL = os.getenv("FROM_EMAIL")
if not FROM_EMAIL:
    sys.exit("FROM_EMAIL is required but not set")

TO_EMAIL = os.getenv("TO_EMAIL")
if not TO_EMAIL:
    sys.exit("TO_EMAIL is required but not set")

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
if not SENDGRID_API_KEY:
    sys.exit("SENDGRID_API_KEY is required but not set")

HOW_MANY_SEARCHES = 3
MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")


class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final report in markdown format.")
    follow_up_questions: list[str] = Field(description="Suggested topics to research further.")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _sg_client() -> sendgrid.SendGridAPIClient:
    return sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)


@function_tool
def send_email(subject: str, html_body: str) -> dict[str, str]:
    """Send an HTML email with the given subject and body."""
    mail = Mail(Email(FROM_EMAIL), To(TO_EMAIL), subject, Content("text/html", html_body)).get()
    _sg_client().client.mail.send.post(request_body=mail)
    return {"status": "success"}


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

search_agent = Agent(
    name="Search agent",
    instructions=(
        "You are a research assistant. Given a search term, search the web and produce a concise summary "
        "of the results. The summary must be 2-3 paragraphs and under 300 words. Capture the main points. "
        "Write succinctly — no need for complete sentences or perfect grammar. This will be consumed by "
        "someone synthesizing a report, so capture the essence and ignore fluff. Return only the summary."
    ),
    tools=[WebSearchTool(search_context_size="low")],
    model=MODEL,
    model_settings=ModelSettings(tool_choice="required"),
)

planner_agent = Agent(
    name="Planner agent",
    instructions=(
        f"You are a helpful research assistant. Given a query, come up with a set of web searches "
        f"to perform to best answer the query. Output {HOW_MANY_SEARCHES} search terms."
    ),
    model=MODEL,
    output_type=WebSearchPlan,
)

writer_agent = Agent(
    name="Writer agent",
    instructions=(
        "You are a senior researcher tasked with writing a cohesive report for a research query. "
        "You will be provided with the original query and initial research from a research assistant.\n"
        "First outline the report structure, then write the full report as your final output.\n"
        "The output must be in markdown format, detailed and lengthy — aim for 5-10 pages, at least 1000 words."
    ),
    model=MODEL,
    output_type=ReportData,
)

email_agent = Agent(
    name="Email agent",
    instructions=(
        "You are able to send a nicely formatted HTML email based on a detailed report. "
        "You will be provided with a markdown report. Use your send_email tool to send one email, "
        "converting the report into clean, well-presented HTML with an appropriate subject line."
    ),
    tools=[send_email],
    model=MODEL,
)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

async def plan_searches(query: str) -> WebSearchPlan:
    print("Planning searches...")
    result = await Runner.run(planner_agent, f"Query: {query}")
    print(f"Will perform {len(result.final_output.searches)} searches")
    return result.final_output


async def perform_searches(search_plan: WebSearchPlan) -> list[str]:
    print("Searching...")
    tasks = [asyncio.create_task(_run_search(item)) for item in search_plan.searches]
    results = await asyncio.gather(*tasks)
    print("Finished searching")
    return list(results)


async def _run_search(item: WebSearchItem) -> str:
    prompt = f"Search term: {item.query}\nReason for searching: {item.reason}"
    result = await Runner.run(search_agent, prompt)
    return result.final_output


async def write_report(query: str, search_results: list[str]) -> ReportData:
    print("Writing report...")
    prompt = f"Original query: {query}\nSummarized search results: {search_results}"
    result = await Runner.run(writer_agent, prompt)
    print("Finished writing report")
    return result.final_output


async def email_report(report: ReportData) -> None:
    print("Sending email...")
    await Runner.run(email_agent, report.markdown_report)
    print("Email sent")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    query = "Latest AI Agent frameworks in 2025"

    with trace("Research trace"):
        print("Starting research...")
        search_plan = await plan_searches(query)
        search_results = await perform_searches(search_plan)
        report = await write_report(query, search_results)
        await email_report(report)

    print(f"\nSummary:\n{report.short_summary}")
    print("\nFollow-up questions:")
    for question in report.follow_up_questions:
        print(f"  - {question}")
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
