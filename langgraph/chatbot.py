from typing import Annotated, TypedDict

import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv(override=True)


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(model="gpt-4o-mini")


def chatbot_node(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()


def chat(user_input: str, history: list) -> str:
    messages = history + [{"role": "user", "content": user_input}]
    result = graph.invoke({"messages": messages})
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()
