import os
import uuid
import asyncio
import psutil
from typing import Annotated, Any, Dict, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import openai

from sidekick_tools import other_tools, playwright_tools

load_dotenv(override=True)

MAX_ITERATIONS = 10
CONVERSATION_CHAR_LIMIT = 8000


# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    plan: Optional[str]
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool
    iteration: int


# ── Structured outputs ────────────────────────────────────────────────────────

class PlannerOutput(BaseModel):
    plan: str = Field(description="Numbered step-by-step plan to accomplish the task")


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Specific feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if the assistant has a question, needs clarification, or is stuck"
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

class Sidekick:
    def __init__(self):
        self.tools: List[Any] = []
        self._static_tools: List[Any] = []
        self._worker_llm: Optional[ChatOpenAI] = None       # base LLM — reused across browser reinits
        self.worker_llm_with_tools = None
        self.planner_llm = None
        self.evaluator_llm = None
        self.graph = None
        self._recursion_limit: int = 1 + MAX_ITERATIONS * 8
        self.sidekick_id = str(uuid.uuid4())
        self.memory = MemorySaver()
        self.browser = None
        self.playwright = None

    async def setup(self):
        browser_tools, self.browser, self.playwright = await playwright_tools()
        self._static_tools = other_tools()
        self.tools = browser_tools + self._static_tools

        # Store base worker LLM so _reinit_browser can rebind without recreating it
        self._worker_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.worker_llm_with_tools = self._worker_llm.bind_tools(self.tools)
        self.planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(PlannerOutput)
        self.evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(EvaluatorOutput)

        self._build_graph()

    async def _reinit_browser(self):
        """Re-open Chromium and rebind tools. Called lazily at the start of run()
        when the browser was closed after the previous task completed.
        LLMs are reused (stateless). MemorySaver is reused — history is preserved."""
        browser_tools, self.browser, self.playwright = await playwright_tools()
        self.tools = browser_tools + self._static_tools
        # Rebind existing LLM instance to new tool set — no need to recreate the LLM
        self.worker_llm_with_tools = self._worker_llm.bind_tools(self.tools)
        self._build_graph()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    async def planner(self, state: State) -> Dict[str, Any]:
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
        )
        tool_names = [t.name for t in self.tools]

        system = (
            "You are a strategic planner for an AI agent. Given a task and available tools, "
            "create a clear numbered step-by-step plan. Be specific about which tools to use at each step. "
            "Keep the plan concise (3-7 steps). Focus on the most efficient path to success."
        )
        user = (
            f"Task: {last_human}\n\n"
            f"Success criteria: {state['success_criteria']}\n\n"
            f"Available tools: {', '.join(tool_names)}\n\n"
            "Create a step-by-step plan."
        )

        result = await self.planner_llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        return {"plan": result.plan, "iteration": 0}

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def worker(self, state: State) -> Dict[str, Any]:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        plan_section = f"\nYour execution plan:\n{state['plan']}\n" if state.get("plan") else ""
        feedback_section = (
            f"\nPreviously your response was rejected. Evaluator feedback:\n"
            f"{state['feedback_on_work']}\nAdjust your approach and continue."
            if state.get("feedback_on_work") else ""
        )

        system = (
            f"You are a capable autonomous agent that completes tasks using tools.\n"
            f"Current date and time: {now}"
            f"{plan_section}"
            f"\nSuccess criteria: {state['success_criteria']}\n\n"
            "Work methodically through your plan. Use save_note to record intermediate results.\n"
            "If you have a question for the user, prefix it clearly with \"Question:\".\n"
            "Otherwise, deliver your final answer without asking questions."
            f"{feedback_section}"
        )

        # Replace existing SystemMessage rather than appending a second one
        has_system = any(isinstance(m, SystemMessage) for m in state["messages"])
        if has_system:
            messages = [
                SystemMessage(content=system) if isinstance(m, SystemMessage) else m
                for m in state["messages"]
            ]
        else:
            messages = [SystemMessage(content=system)] + list(state["messages"])

        response = await self.worker_llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    async def evaluator(self, state: State) -> Dict[str, Any]:
        last_response = state["messages"][-1].content or ""
        conversation = _format_conversation(state["messages"])

        # Truncate to avoid token overflow on long agentic runs
        if len(conversation) > CONVERSATION_CHAR_LIMIT:
            conversation = "...[truncated]\n" + conversation[-CONVERSATION_CHAR_LIMIT:]

        iteration = state.get("iteration", 0)
        max_reached_note = (
            "NOTE: Maximum iterations reached — accept the best available answer.\n\n"
            if iteration >= MAX_ITERATIONS else ""
        )

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system = (
            f"You are a strict evaluator. The current date and time is {now}. "
            "The assistant has access to real-time tools (datetime, calculator, web search, etc.). "
            "When the assistant's answer comes from a tool call, treat that result as ground truth — "
            "do NOT override it with your own knowledge of dates, prices, or other real-world facts. "
            "Assess whether the response fully meets the success criteria. "
            "Be demanding — only approve if the criteria are genuinely satisfied. "
            "Identify any gaps or questions the assistant raised."
        )
        user = (
            f"Conversation:\n{conversation}\n\n"
            f"Success criteria: {state['success_criteria']}\n\n"
            f"Plan that was followed:\n{state.get('plan', 'No plan available')}\n\n"
            f"Final assistant response to evaluate:\n{last_response}\n\n"
            f"Prior evaluator feedback (if any): {state.get('feedback_on_work') or 'None'}\n\n"
            f"{max_reached_note}"
            "Evaluate strictly. If the assistant asked a question, set user_input_needed=True."
        )

        result = await self.evaluator_llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])

        return {
            # Use AIMessage for type consistency with add_messages reducer
            "messages": [AIMessage(content=f"[Evaluator] {result.feedback}")],
            "feedback_on_work": result.feedback,
            "success_criteria_met": result.success_criteria_met,
            "user_input_needed": result.user_input_needed,
            "iteration": iteration + 1,
        }

    # ── Routers ───────────────────────────────────────────────────────────────

    def worker_router(self, state: State) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "evaluator"

    def evaluation_router(self, state: State) -> str:
        if (
            state["success_criteria_met"]
            or state["user_input_needed"]
            or state.get("iteration", 0) >= MAX_ITERATIONS
        ):
            return "END"
        return "worker"

    # ── Graph ─────────────────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(State)

        g.add_node("planner", self.planner)
        g.add_node("worker", self.worker)
        g.add_node("tools", ToolNode(tools=self.tools))
        g.add_node("evaluator", self.evaluator)

        g.add_edge(START, "planner")
        g.add_edge("planner", "worker")
        g.add_conditional_edges("worker", self.worker_router, {"tools": "tools", "evaluator": "evaluator"})
        g.add_edge("tools", "worker")
        g.add_conditional_edges("evaluator", self.evaluation_router, {"worker": "worker", "END": END})

        self.graph = g.compile(checkpointer=self.memory)

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, message: str, success_criteria: str, history: list) -> tuple[list, str]:
        # Lazy browser re-init: browser is closed after each task to free resources.
        if self.browser is None:
            await self._reinit_browser()

        config = {
            "configurable": {"thread_id": self.sidekick_id},
            "recursion_limit": self._recursion_limit,
        }
        state = {
            "messages": [HumanMessage(content=message)],
            "success_criteria": success_criteria or "The answer is clear, complete, and accurate.",
            "plan": None,
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "iteration": 0,
        }

        try:
            result = await self.graph.ainvoke(state, config=config)
        finally:
            # Auto-close Chromium after every task so no browser window lingers.
            # Next run() call will re-open it lazily via _reinit_browser().
            await self.cleanup()

        worker_reply = _last_worker_reply(result["messages"])
        last_msg = result["messages"][-1]
        evaluator_note = last_msg.content or ""
        plan = result.get("plan", "")

        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": worker_reply},
            {"role": "assistant", "content": evaluator_note},
        ]
        return updated_history, plan

    async def cleanup(self):
        try:
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            # Give processes a moment to exit gracefully
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Browser cleanup error: {e}")
        finally:
            # Force-kill any lingering Playwright driver or Chromium child processes
            _kill_playwright_children()
            self.browser = None
            self.playwright = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kill_playwright_children() -> None:
    """Force-kill Playwright driver (node) and Chromium child processes
    that may linger after playwright.stop() returns on macOS."""
    try:
        me = psutil.Process(os.getpid())
        for child in me.children(recursive=True):
            cmd = " ".join(child.cmdline())
            if "playwright" in cmd or "chromium" in cmd.lower() or "Chromium" in cmd:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except Exception:
        pass


def _last_worker_reply(messages: List[Any]) -> str:
    """Walk backwards to find the last worker AIMessage (not an evaluator note)."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.content.startswith("[Evaluator]"):
            return msg.content
    return ""


def _format_conversation(messages: List[Any]) -> str:
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)
