import asyncio
import logging
import os

from agents import Agent, Runner, trace
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PLAYWRIGHT_PARAMS = {"command": "npx", "args": ["-y", "@playwright/mcp@0.0.76"]}

INSTRUCTIONS = """
You browse the internet to accomplish your instructions.
You are highly capable at browsing the internet independently to accomplish your task,
including accepting all cookies and clicking 'not now' as appropriate to get to the content you need.
If one website isn't fruitful, try another. Be persistent until you have solved your assignment,
trying different options and sites as needed.
When you need to write files, you do that inside the sandbox folder only.
"""


async def main() -> None:
    sandbox_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sandbox"))
    os.makedirs(sandbox_path, exist_ok=True)

    files_params = {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem@2026.1.14", sandbox_path],
    }

    logger.info("Starting MCP servers, sandbox: %s", sandbox_path)
    try:
        async with MCPServerStdio(params=files_params, client_session_timeout_seconds=60) as mcp_files:
            async with MCPServerStdio(params=PLAYWRIGHT_PARAMS, client_session_timeout_seconds=60) as mcp_browser:
                agent = Agent(
                    name="investigator",
                    instructions=INSTRUCTIONS,
                    model="gpt-4.1-mini",
                    mcp_servers=[mcp_files, mcp_browser],
                )
                with trace("investigate"):
                    logger.info("Agent started")
                    result = await Runner.run(
                        agent,
                        "Find a great recipe for Banoffee Pie, then summarize it in markdown to banoffee.md",
                        max_turns=20,
                    )
                    logger.info("Agent finished")
                    print(result.final_output)
    except Exception:
        logger.exception("Agent run failed")
        raise


if __name__ == "__main__":
    asyncio.run(main())