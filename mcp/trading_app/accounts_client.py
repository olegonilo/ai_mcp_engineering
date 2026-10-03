import json

import mcp
from agents import FunctionTool
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_params import trader_mcp_server_params


def _params(name: str) -> StdioServerParameters:
    # Same launch spec as the trader's own accounts server (absolute path, cwd, forwarded secrets).
    return StdioServerParameters(**trader_mcp_server_params(name)[0])


async def list_accounts_tools(name: str):
    async with stdio_client(_params(name)) as streams:
        async with mcp.ClientSession(*streams) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            return tools_result.tools


async def call_accounts_tool(name: str, tool_name: str, tool_args: dict):
    async with stdio_client(_params(name)) as streams:
        async with mcp.ClientSession(*streams) as session:
            await session.initialize()
            return await session.call_tool(tool_name, tool_args)


async def read_accounts_resource(name: str) -> str:
    async with stdio_client(_params(name)) as streams:
        async with mcp.ClientSession(*streams) as session:
            await session.initialize()
            result = await session.read_resource(f"accounts://accounts_server/{name}")
            return result.contents[0].text


async def read_strategy_resource(name: str) -> str:
    async with stdio_client(_params(name)) as streams:
        async with mcp.ClientSession(*streams) as session:
            await session.initialize()
            result = await session.read_resource(f"accounts://strategy/{name}")
            return result.contents[0].text


async def get_accounts_tools_openai(name: str) -> list[FunctionTool]:
    openai_tools = []
    for tool in await list_accounts_tools(name):
        schema = {**tool.inputSchema, "additionalProperties": False}
        openai_tool = FunctionTool(
            name=tool.name,
            description=tool.description,
            params_json_schema=schema,
            on_invoke_tool=lambda ctx, args, toolname=tool.name: call_accounts_tool(name, toolname, json.loads(args)),
        )
        openai_tools.append(openai_tool)
    return openai_tools
