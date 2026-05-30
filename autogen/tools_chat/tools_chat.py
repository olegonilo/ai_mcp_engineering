import asyncio
import textwrap
from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from typing import Literal

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken, Image as AGImage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.langchain import LangChainToolAdapter
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool

load_dotenv(override=True)

IMAGE_URL = "https://images2.alphacoders.com/136/thumb-1920-1366761.jpeg"
FLIGHT_PROMPT = "Find a one-way non-stop flight from Bratislava to Bangkok in July 2026."
SANDBOX_DIR = Path(__file__).parent / "sandbox"  # anchored to script dir, not CWD


class ImageDescription(BaseModel):
    scene: str = Field(description="Briefly, the overall scene of the image")
    message: str = Field(description="The point that the image is trying to convey")
    style: str = Field(description="The artistic style of the image")
    orientation: Literal["portrait", "landscape", "square"] = Field(description="The orientation of the image")


def load_image(url: str) -> MultiModalMessage:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        pil_image = Image.open(BytesIO(response.content))
    except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"Failed to load image from {url}: {exc}") from exc
    return MultiModalMessage(content=[AGImage(pil_image)], source="user")


def make_serper_tool() -> LangChainToolAdapter:
    serper = GoogleSerperAPIWrapper()
    langchain_tool = Tool(
        name="internet_search",
        func=serper.run,
        description="Useful for when you need to search the internet",
    )
    return LangChainToolAdapter(langchain_tool)


def make_file_tools() -> list:
    SANDBOX_DIR.mkdir(exist_ok=True)
    return [LangChainToolAdapter(t) for t in FileManagementToolkit(root_dir=str(SANDBOX_DIR)).get_tools()]


async def demo_image_description(client: OpenAIChatCompletionClient) -> None:
    """Basic image description."""
    multi_modal_message = load_image(IMAGE_URL)
    agent = AssistantAgent(
        name="description_agent",
        model_client=client,
        system_message="You are good at describing images",
    )
    response = await agent.on_messages([multi_modal_message], cancellation_token=CancellationToken())
    print("=== Image Description ===")
    print(response.chat_message.content)


async def demo_structured_description(client: OpenAIChatCompletionClient) -> None:
    """Structured image description using Pydantic output."""
    multi_modal_message = load_image(IMAGE_URL)
    agent = AssistantAgent(
        name="description_agent",
        model_client=client,
        system_message="You are good at describing images in detail",
        output_content_type=ImageDescription,
    )
    response = await agent.on_messages([multi_modal_message], cancellation_token=CancellationToken())
    reply = response.chat_message.content
    print("=== Structured Description ===")
    if not isinstance(reply, ImageDescription):
        print(f"[warn] expected ImageDescription, got {type(reply).__name__}: {reply}")
        return
    print(f"Scene:\n{textwrap.fill(reply.scene)}\n")
    print(f"Message:\n{textwrap.fill(reply.message)}\n")
    print(f"Style:\n{textwrap.fill(reply.style)}\n")
    print(f"Orientation:\n{reply.orientation}\n")


async def demo_tool_agent(client: OpenAIChatCompletionClient) -> None:
    """Single agent with search + file tools."""
    tools = [make_serper_tool(), *make_file_tools()]
    for tool in tools:
        print(f"  Tool: {tool.name} — {tool.description[:60]}")

    agent = AssistantAgent(
        name="searcher",
        model_client=client,
        tools=tools,
        reflect_on_tool_use=True,
    )
    prompt = (
        f"{FLIGHT_PROMPT}\n"
        "First search online for promising deals. "
        "Next, write all the deals to a file called flights.md with full details. "
        "Finally, select the one you think is best and reply with a short summary. "
        "Reply with the selected flight only, and only after you have written the details to the file."
    )
    result = await agent.on_messages(
        [TextMessage(content=prompt, source="user")],
        cancellation_token=CancellationToken(),
    )
    print("=== Tool Agent ===")
    for msg in result.inner_messages or []:
        print(msg.content)
    print(result.chat_message.content)

    # Follow-up turn to trigger file writing if needed
    result = await agent.on_messages(
        [TextMessage(content="OK proceed", source="user")],
        cancellation_token=CancellationToken(),
    )
    for msg in result.inner_messages or []:
        print(msg.content)
    print(result.chat_message.content)


async def demo_multi_agent_team(client: OpenAIChatCompletionClient) -> None:
    """Researcher + evaluator team with APPROVE termination."""
    primary = AssistantAgent(
        "primary",
        model_client=client,
        tools=[make_serper_tool()],
        system_message="You are a helpful AI research assistant who looks for promising flight deals. Incorporate any feedback you receive.",
    )
    evaluator = AssistantAgent(
        "evaluator",
        model_client=client,
        system_message="Provide constructive feedback. Respond with 'APPROVE' when your feedback is addressed.",
    )
    team = RoundRobinGroupChat(
        [primary, evaluator],
        termination_condition=TextMentionTermination("APPROVE"),
        max_turns=20,
    )
    result = await team.run(task=FLIGHT_PROMPT)
    print("=== Multi-Agent Team ===")
    for msg in result.messages:
        print(f"{msg.source}:\n{msg.content}\n")


async def demo_mcp_fetch(client: OpenAIChatCompletionClient) -> None:
    """Agent using MCP fetch server to summarize a website."""
    try:
        fetch_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-fetch"], read_timeout_seconds=60)
        fetcher_tools = await mcp_server_tools(fetch_mcp_server)
    except Exception as exc:
        print(f"=== MCP Fetch Agent ===\n[error] MCP server unavailable: {exc}")
        return
    agent = AssistantAgent(
        name="fetcher",
        model_client=client,
        tools=fetcher_tools,  # type: ignore[arg-type]
        reflect_on_tool_use=True,
    )
    result = await agent.run(task="Fetch https://www.expedition33.com and summarize what you learn. Reply in Markdown.")
    print("=== MCP Fetch Agent ===")
    print(result.messages[-1].content if result.messages else "[no response]")


async def main() -> None:
    client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    try:
        await demo_image_description(client)
        await demo_structured_description(client)
        await demo_tool_agent(client)
        await demo_multi_agent_team(client)
        await demo_mcp_fetch(client)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
