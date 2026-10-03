import asyncio
import logging
from contextlib import AsyncExitStack
from functools import cache

from agents import Agent, OpenAIChatCompletionsModel, Runner, Tool, trace
from agents.mcp import MCPServerStdio
from openai import AsyncOpenAI

from accounts import Account
from config import MAX_TURNS, MCP_SESSION_TIMEOUT_SECONDS, TRADER_RUN_TIMEOUT_SECONDS, require_env
from market import PriceUnavailableError
from mcp_params import researcher_mcp_server_params, trader_mcp_server_params
from templates import (
    rebalance_message,
    research_tool,
    researcher_instructions,
    trade_message,
    trader_instructions,
)
from tracers import make_trace_id

logger = logging.getLogger(__name__)

# provider -> (base_url, env var holding its API key)
PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "grok": ("https://api.x.ai/v1", "GROK_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY"),
}


@cache
def _client(provider: str) -> AsyncOpenAI:
    # The key is always passed explicitly: with api_key=None the SDK silently falls back to
    # OPENAI_API_KEY and would send the OpenAI secret to a third-party base_url.
    base_url, key_env = PROVIDERS[provider]
    return AsyncOpenAI(base_url=base_url, api_key=require_env(key_env))


def get_model(model_name: str):
    if "/" in model_name:
        provider = "openrouter"
    elif "deepseek" in model_name:
        provider = "deepseek"
    elif "grok" in model_name:
        provider = "grok"
    elif "gemini" in model_name:
        provider = "gemini"
    else:
        return model_name
    return OpenAIChatCompletionsModel(model=model_name, openai_client=_client(provider))


def get_researcher(mcp_servers, model_name) -> Agent:
    return Agent(
        name="Researcher",
        instructions=researcher_instructions(),
        model=get_model(model_name),
        mcp_servers=mcp_servers,
    )


def get_researcher_tool(mcp_servers, model_name) -> Tool:
    researcher = get_researcher(mcp_servers, model_name)
    return researcher.as_tool(tool_name="Researcher", tool_description=research_tool())


class Trader:
    def __init__(self, name: str, lastname="Trader", model_name="gpt-4o-mini"):
        self.name = name
        self.lastname = lastname
        self.agent = None
        self.model_name = model_name
        self.do_trade = True

    def create_agent(self, trader_mcp_servers, researcher_mcp_servers) -> Agent:
        tool = get_researcher_tool(researcher_mcp_servers, self.model_name)
        self.agent = Agent(
            name=self.name,
            instructions=trader_instructions(self.name),
            model=get_model(self.model_name),
            tools=[tool],
            mcp_servers=trader_mcp_servers,
        )
        return self.agent

    def get_account_report(self) -> str:
        # Read the shared database directly instead of spawning two extra MCP server processes.
        return Account.get(self.name).report(record=True)

    async def run_agent(self, trader_mcp_servers, researcher_mcp_servers):
        self.agent = self.create_agent(trader_mcp_servers, researcher_mcp_servers)
        account = await asyncio.to_thread(self.get_account_report)
        strategy = await asyncio.to_thread(lambda: Account.get(self.name).get_strategy())
        message = (
            trade_message(self.name, strategy, account)
            if self.do_trade
            else rebalance_message(self.name, strategy, account)
        )
        await Runner.run(self.agent, message, max_turns=MAX_TURNS)

    async def run_with_mcp_servers(self):
        async with AsyncExitStack() as stack:
            trader_mcp_servers = [
                await stack.enter_async_context(
                    MCPServerStdio(params, client_session_timeout_seconds=MCP_SESSION_TIMEOUT_SECONDS)
                )
                for params in trader_mcp_server_params(self.name)
            ]
            researcher_mcp_servers = [
                await stack.enter_async_context(
                    MCPServerStdio(params, client_session_timeout_seconds=MCP_SESSION_TIMEOUT_SECONDS)
                )
                for params in researcher_mcp_server_params(self.name)
            ]
            await self.run_agent(trader_mcp_servers, researcher_mcp_servers)

    async def run_with_trace(self):
        trace_name = f"{self.name}-trading" if self.do_trade else f"{self.name}-rebalancing"
        trace_id = make_trace_id(self.name.lower())
        with trace(trace_name, trace_id=trace_id):
            await self.run_with_mcp_servers()

    async def run(self) -> None:
        try:
            async with asyncio.timeout(TRADER_RUN_TIMEOUT_SECONDS):
                await self.run_with_trace()
        except TimeoutError:
            logger.error("Trader %s timed out after %ss", self.name, TRADER_RUN_TIMEOUT_SECONDS)
            return
        except Exception:
            logger.exception("Error running trader %s", self.name)
            return
        # Alternate between trading and rebalancing only after a completed run.
        self.do_trade = not self.do_trade
        try:
            await asyncio.to_thread(Account.get(self.name).record_portfolio_value)
        except PriceUnavailableError as e:
            logger.warning("Could not record portfolio value for %s: %s", self.name, e)
