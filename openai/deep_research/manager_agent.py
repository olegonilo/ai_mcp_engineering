from agents import Agent
from email_agent import email_agent
from planner_agent import planner_agent
from search_agent import search_agent
from writer_agent import writer_agent

INSTRUCTIONS = """
You are a Research Manager. Given a research query with user clarifications, run the full pipeline:

1. PLAN: Call the planner tool with the full input (query + clarifications). It returns a JSON object
   with a "searches" list; each item has a "query" string and a "reason" string.
2. SEARCH: Extract the "query" field from each item in "searches" and call the search tool once per query.
   Run ALL of them before moving on.
3. WRITE: Call the writer tool with the original query, clarifications, and all search results combined.
4. EMAIL: Hand off to the email agent to format and send the final report.

Important: run ALL searches before writing. Pass all results to the writer.
"""

manager_agent = Agent(
    name="Research Manager",
    instructions=INSTRUCTIONS,
    tools=[
        planner_agent.as_tool(
            tool_name="planner",
            tool_description=(
                "Plan web searches for the research query, taking user clarifications into account. "
                "Returns a JSON object with a 'searches' list; each item has a 'query' string to search for."
            ),
        ),
        search_agent.as_tool(
            tool_name="search",
            tool_description="Search the web for a specific term and return a concise summary of the results.",
        ),
        writer_agent.as_tool(
            tool_name="writer",
            tool_description="Write a detailed research report from the original query and all search results.",
        ),
    ],
    handoffs=[email_agent],
    model="gpt-4o",
)
