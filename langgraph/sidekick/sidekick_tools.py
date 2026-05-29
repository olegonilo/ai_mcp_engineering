import os
import math
import re
import shlex
import subprocess
import requests as http_requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from langchain.agents import Tool
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit, FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_experimental.tools import PythonREPLTool

load_dotenv(override=True)

pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

# Lazy init — crashes at import if key is missing without this guard
_serper_key = os.getenv("SERPER_API_KEY")
serper = GoogleSerperAPIWrapper() if _serper_key else None

# Anchor to this file's directory so sandbox paths are correct regardless of CWD
SANDBOX_DIR = str(Path(__file__).parent / "sandbox")
NOTES_FILE = os.path.join(SANDBOX_DIR, "agent_notes.txt")


# ── Browser ───────────────────────────────────────────────────────────────────

async def playwright_tools():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright


# ── Tool implementations ──────────────────────────────────────────────────────

def push_notification(text: str) -> str:
    if not pushover_token or not pushover_user:
        return "Push notification failed: PUSHOVER_TOKEN or PUSHOVER_USER not configured."
    try:
        resp = http_requests.post(
            pushover_url,
            data={"token": pushover_token, "user": pushover_user, "message": text},
            timeout=5,
        )
        resp.raise_for_status()
        return "Push notification sent."
    except Exception as e:
        return f"Push notification failed: {e}"


def get_current_datetime(_: str = "") -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    try:
        safe_env = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_env.update({"abs": abs, "round": round, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, safe_env)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


def fetch_url(url: str) -> str:
    try:
        resp = http_requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.text[:5000]
    except Exception as e:
        return f"Failed to fetch URL: {e}"


_BLOCKED_CMDS = {
    "rm", "rmdir", "mv", "cp", "sudo", "chmod", "chown",
    "kill", "pkill", "shutdown", "reboot", "dd", "mkfs",
    "fdisk", "bash", "sh", "zsh", "fish", "eval", "exec",
}
# Detect shell injection: semicolons, pipes, backticks, $() substitutions
_INJECTION_RE = re.compile(r'[;&|`]|\$\(')


def run_shell(command: str) -> str:
    if _INJECTION_RE.search(command):
        return "Blocked: shell injection patterns detected (;, |, &, $(), ` )."
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        return f"Shell parse error: {e}"
    if not tokens:
        return "(empty command)"
    first_cmd = os.path.basename(tokens[0])
    if first_cmd in _BLOCKED_CMDS:
        return f"Blocked: '{first_cmd}' is a restricted command."
    try:
        # shell=False prevents injection via shell metacharacters
        result = subprocess.run(tokens, capture_output=True, text=True, timeout=30)
        output = (result.stdout + result.stderr).strip()
        return output[:3000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds."
    except FileNotFoundError:
        return f"Command not found: {tokens[0]}"
    except Exception as e:
        return f"Shell error: {e}"


def save_note(content: str) -> str:
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    ts = datetime.now().strftime("%H:%M:%S")
    with open(NOTES_FILE, "a") as f:
        f.write(f"[{ts}] {content}\n")
    return "Note saved."


def read_notes(_: str = "") -> str:
    try:
        with open(NOTES_FILE) as f:
            content = f.read().strip()
        return content or "No notes yet."
    except FileNotFoundError:
        return "No notes yet."


def clear_notes(_: str = "") -> str:
    try:
        os.remove(NOTES_FILE)
        return "Notes cleared."
    except FileNotFoundError:
        return "Nothing to clear."


# ── Tool assembly ─────────────────────────────────────────────────────────────

def _file_tools():
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    return FileManagementToolkit(root_dir=SANDBOX_DIR).get_tools()


def other_tools():
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
    python_repl = PythonREPLTool()

    tools = _file_tools() + [
        Tool(name="send_push_notification", func=push_notification,
             description="Send a push notification to the user when a task completes or needs attention."),
        Tool(name="get_current_datetime", func=get_current_datetime,
             description="Get the current date and time as a formatted string."),
        Tool(name="calculator", func=calculate,
             description="Evaluate a math expression. Supports +,-,*,/,**,sqrt,sin,cos,log,etc. Example: 'sqrt(144)+2**8'"),
        Tool(name="fetch_url", func=fetch_url,
             description="Fetch raw content from a URL. Returns first 5000 characters. Use for APIs or direct page content."),
        Tool(name="run_shell_command", func=run_shell,
             description="Run a shell command (ls, cat, grep, find, wc, etc.). Shell injection and destructive commands are blocked."),
        Tool(name="save_note", func=save_note,
             description="Save a note or intermediate result to working memory. Useful for tracking sub-task progress."),
        Tool(name="read_notes", func=read_notes,
             description="Read all notes saved to working memory during this session."),
        Tool(name="clear_notes", func=clear_notes,
             description="Clear all working memory notes to start fresh on a new approach."),
        wiki,
        arxiv,
        python_repl,
    ]

    if serper:
        tools.append(Tool(
            name="search_web", func=serper.run,
            description="Search the web for current information, news, facts, or recent events.",
        ))

    return tools
