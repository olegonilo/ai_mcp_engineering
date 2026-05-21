import atexit
import asyncio
import logging
import os
import uuid
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import gradio as gr
import nest_asyncio
from dotenv import load_dotenv
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REQUIRED_ENV = ["OPENAI_API_KEY"]
_missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
if _missing:
    raise EnvironmentError(f"Missing required environment variables: {_missing}")

_HEADLESS = os.environ.get("BROWSER_HEADLESS", "false").lower() != "false"
_DEBUG = os.environ.get("SIDEKICK_DEBUG", "").lower() in ("1", "true")
_MAX_ITERATIONS = 10


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool


# ---------------------------------------------------------------------------
# Browser state
# Initialised twice:
#   1. Bootstrap at module load (headless=True) — only to extract tool schemas.
#   2. demo.load() calls _reinit_browser — real browser in Gradio's event loop.
# ---------------------------------------------------------------------------
_playwright_handle: Optional[Any] = None
async_browser: Optional[Any] = None
_browser_lock = asyncio.Lock()
_browser_in_gradio_loop = False  # True after _reinit_browser runs inside Gradio's event loop


async def _create_browser(headless: bool) -> None:
    global _playwright_handle, async_browser
    _playwright_handle = await async_playwright().start()
    async_browser = await _playwright_handle.chromium.launch(headless=headless)


# Bootstrap so PlayWrightBrowserToolkit can extract tool schemas at import time.
nest_asyncio.apply()
asyncio.get_event_loop().run_until_complete(_create_browser(headless=True))

toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()

worker_llm_with_tools = ChatOpenAI(model="gpt-5.4-mini").bind_tools(tools)
evaluator_llm_with_output = ChatOpenAI(model="gpt-5.4-mini").with_structured_output(EvaluatorOutput)


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

async def worker(state: State) -> Dict[str, Any]:
    system_prompt = f"""You are a helpful assistant that can use tools to complete tasks.
You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
This is the success criteria:
{state['success_criteria']}
You should reply either with a question for the user about this assignment, or with your final response.
If you have a question for the user, you need to reply by clearly stating your question. An example might be:

Question: please clarify whether you want a summary or a detailed answer

If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer."""

    if state.get("feedback_on_work"):
        system_prompt += f"""

Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
Here is the feedback on why this was rejected:
{state['feedback_on_work']}
With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""

    non_system = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    messages = [SystemMessage(content=system_prompt)] + non_system

    response = await worker_llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


def worker_router(state: State) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "evaluator"


def format_conversation(messages: List[Any]) -> str:
    lines = ["Conversation history:\n"]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"User: {content}")
        elif isinstance(msg, AIMessage):
            if not isinstance(msg.content, str):
                continue
            if msg.content.startswith("Evaluator Feedback:"):
                continue
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                continue  # skip tool-dispatch stubs; they carry no user-visible content
            if msg.content:
                lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)


async def evaluator(state: State) -> Dict[str, Any]:
    last_response = state["messages"][-1].content
    feedback_on_work = state.get("feedback_on_work")

    system_message = (
        "You are an evaluator that determines if a task has been completed successfully by an Assistant. "
        "Assess the Assistant's last response based on the given criteria. Respond with your feedback, "
        "and with your decision on whether the success criteria has been met, "
        "and whether more input is needed from the user."
    )

    user_message = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take based on the last response from the Assistant.

The entire conversation with the assistant, with the user's original request and all replies, is:
{format_conversation(state['messages'])}

The success criteria for this assignment is:
{state['success_criteria']}

And the final response from the Assistant that you are evaluating is:
{last_response}

Respond with your feedback, and decide if the success criteria is met by this response.
Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help."""

    if feedback_on_work:
        user_message += (
            f"\nAlso, note that in a prior attempt from the Assistant, you provided this feedback: {feedback_on_work}\n"
            "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."
        )

    eval_result = await evaluator_llm_with_output.ainvoke([
        SystemMessage(content=system_message),
        HumanMessage(content=user_message),
    ])

    return {
        "messages": [{"role": "assistant", "content": f"Evaluator Feedback: {eval_result.feedback}"}],
        "feedback_on_work": eval_result.feedback,
        "success_criteria_met": eval_result.success_criteria_met,
        "user_input_needed": eval_result.user_input_needed,
    }


def route_based_on_evaluation(state: State) -> str:
    if state["success_criteria_met"] or state["user_input_needed"]:
        return "END"
    return "worker"


graph_builder = StateGraph(State)
graph_builder.add_node("worker", worker)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_node("evaluator", evaluator)
graph_builder.add_conditional_edges("worker", worker_router, {"tools": "tools", "evaluator": "evaluator"})
graph_builder.add_edge("tools", "worker")
graph_builder.add_conditional_edges("evaluator", route_based_on_evaluation, {"worker": "worker", "END": END})
graph_builder.add_edge(START, "worker")

graph = graph_builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Browser lifecycle
# ---------------------------------------------------------------------------

async def _reinit_browser() -> None:
    """(Re-)create the browser inside Gradio's event loop; patch all tool instances."""
    global _playwright_handle, async_browser, _browser_in_gradio_loop
    async with _browser_lock:
        if _browser_in_gradio_loop and _playwright_handle is not None:
            # Safe to close — previous browser was created in this same event loop.
            try:
                await async_browser.close()
                await _playwright_handle.stop()
            except Exception:
                logging.warning("Error closing previous browser", exc_info=True)
        # Bootstrap browser (created at module load in a different loop) is intentionally
        # left open here; its subprocess is cleaned up by _atexit_cleanup on exit.
        _playwright_handle = await async_playwright().start()
        async_browser = await _playwright_handle.chromium.launch(headless=_HEADLESS)
        _browser_in_gradio_loop = True
        for tool in tools:
            if hasattr(tool, "async_browser"):
                tool.async_browser = async_browser
    logging.info("Playwright browser (re)initialized in Gradio event loop")


def _atexit_cleanup() -> None:
    """Best-effort playwright shutdown on process exit."""
    if _playwright_handle is None:
        return
    try:
        asyncio.run(_playwright_handle.stop())
    except Exception:
        pass


atexit.register(_atexit_cleanup)


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------

def make_thread_id() -> str:
    return str(uuid.uuid4())


def _last_worker_reply(messages: List[Any]) -> str:
    """Return the content of the most recent non-evaluator AIMessage."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.content.startswith("Evaluator Feedback:"):
            return msg.content or "No response generated."
    return "No response generated."


async def process_message(message: str, success_criteria: str, history: list, thread: str) -> list:
    if not message.strip():
        return history
    if not success_criteria.strip():
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please provide success criteria before submitting."},
        ]

    config = {
        "configurable": {"thread_id": thread},
        # Budget: up to _MAX_ITERATIONS rounds; each round = worker + up to ~5 tool calls + evaluator ≈ 12 edges
        "recursion_limit": _MAX_ITERATIONS * 12,
    }
    state = {
        "messages": [{"role": "user", "content": message}],
        "success_criteria": success_criteria,
        "feedback_on_work": None,
        "success_criteria_met": False,
        "user_input_needed": False,
    }

    try:
        logging.info("Processing message on thread %s", thread)
        result = await graph.ainvoke(state, config=config)
    except Exception:
        logging.error("Graph invocation failed on thread %s", thread, exc_info=True)
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Sorry, something went wrong. Please try again."},
        ]
    finally:
        await _close_browser_contexts()

    reply = _last_worker_reply(result["messages"])
    entries: List[Dict[str, str]] = [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    if _DEBUG:
        last = result["messages"][-1]
        if isinstance(last, AIMessage) and last.content.startswith("Evaluator Feedback:"):
            entries.append({"role": "assistant", "content": last.content})
    return history + entries


async def _close_browser_contexts() -> None:
    """Close all browser contexts (and their pages) after a task completes."""
    async with _browser_lock:
        if async_browser is None or not _browser_in_gradio_loop:
            return
        for context in list(async_browser.contexts):
            try:
                await context.close()
            except Exception:
                logging.warning("Error closing browser context", exc_info=True)
    logging.info("Browser contexts closed after task")


async def reset() -> tuple:
    async with _browser_lock:
        if async_browser is not None and _browser_in_gradio_loop:
            for context in list(async_browser.contexts):
                await context.close()  # closes child pages implicitly
    return "", "", [], make_thread_id()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Default(primary_hue="emerald")) as demo:
    gr.Markdown("## Sidekick Personal Co-worker")
    thread = gr.State(make_thread_id())

    with gr.Row():
        chatbot = gr.Chatbot(label="Sidekick", height=300, type="messages")
    with gr.Group():
        with gr.Row():
            message = gr.Textbox(show_label=False, placeholder="Your request to your sidekick")
        with gr.Row():
            success_criteria = gr.Textbox(show_label=False, placeholder="What are your success criteria?")
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Go!", variant="primary")

    submit_event = message.submit(process_message, [message, success_criteria, chatbot, thread], [chatbot])
    criteria_event = success_criteria.submit(process_message, [message, success_criteria, chatbot, thread], [chatbot])
    go_event = go_button.click(process_message, [message, success_criteria, chatbot, thread], [chatbot])
    reset_button.click(
        reset, [], [message, success_criteria, chatbot, thread],
        cancels=[submit_event, criteria_event, go_event],
    )
    demo.load(_reinit_browser)

if __name__ == "__main__":
    demo.launch()
