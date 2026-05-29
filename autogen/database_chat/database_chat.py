import asyncio
import os
import sqlite3

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv(override=True)

DB_PATH = "tickets.db"
USER_MESSAGE = TextMessage(content="I'd like to go to London", source="user")


def init_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    cities = [
        ("London", 299),
        ("Paris", 399),
        ("Rome", 499),
        ("Madrid", 550),
        ("Barcelona", 580),
        ("Berlin", 525),
    ]
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE cities (city_name TEXT PRIMARY KEY, round_trip_price REAL)")
        conn.executemany("INSERT INTO cities VALUES (?, ?)", [(c.lower(), p) for c, p in cities])


def get_city_price(city_name: str) -> float | None:
    """Get the roundtrip ticket price to travel to the city."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT round_trip_price FROM cities WHERE city_name = ?", (city_name.lower(),)
        ).fetchone()
    return row[0] if row else None


async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    try:
        init_db()

        # Basic agent — no tools
        basic_agent = AssistantAgent(
            name="airline_agent",
            model_client=model_client,
            system_message="You are a helpful assistant for an airline. You give short, humorous answers.",
            model_client_stream=True,
        )
        response = await basic_agent.on_messages([USER_MESSAGE], CancellationToken())
        print("Basic agent:", response.chat_message.content)

        # Smart agent — with price lookup tool
        smart_agent = AssistantAgent(
            name="smart_airline_agent",
            model_client=model_client,
            system_message="You are a helpful assistant for an airline. You give short, humorous answers, including the price of a roundtrip ticket.",
            model_client_stream=True,
            tools=[get_city_price],
            reflect_on_tool_use=True,
        )
        response = await smart_agent.on_messages([USER_MESSAGE], CancellationToken())
        for msg in (response.inner_messages or []):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            print("Tool call:", content)
        print("Smart agent:", response.chat_message.content)
    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
