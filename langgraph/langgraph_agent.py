import logging
import os
import requests
from typing import Annotated, TypedDict

import gradio as gr
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REQUIRED_ENV = ["OPENAI_API_KEY", "SERPER_API_KEY", "PUSHOVER_TOKEN", "PUSHOVER_USER"]
_missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
if _missing:
    raise EnvironmentError(f"Missing required environment variables: {_missing}")

serper = GoogleSerperAPIWrapper()
pushover_url = "https://api.pushover.net/1/messages.json"


@tool
def search(query: str) -> str:
    """Search the internet for current information."""
    return serper.run(query)


@tool
def send_push_notification(text: str) -> str:
    """Send a push notification to the user."""
    try:
        resp = requests.post(
            pushover_url,
            data={
                "token": os.environ["PUSHOVER_TOKEN"],
                "user": os.environ["PUSHOVER_USER"],
                "message": text,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return "Notification sent successfully."
    except requests.exceptions.RequestException as exc:
        logging.error("Pushover notification failed: %s", exc)
        return f"Failed to send notification: {exc}"


tools = [search, send_push_notification]


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile(checkpointer=MemorySaver())


def chat(user_input: str, _history, request: gr.Request) -> str:
    # LangGraph MemorySaver owns history; one thread per browser session
    config = {"configurable": {"thread_id": request.session_hash}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]}, config=config
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    gr.ChatInterface(chat, type="messages").launch()
