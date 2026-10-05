# AI MCP Engineering

A learning monorepo for agentic AI frameworks: OpenAI Agents SDK, CrewAI, LangGraph, AutoGen, MCP.

## Structure

| Folder | Contents |
|---|---|
| `foundation/` | Core patterns on plain LLM APIs: prompt chaining/parallelization, orchestrator, personal agent with Pushover |
| `openai/` | OpenAI Agents SDK: `deep_research/` (multi-agent research + email), `tools_handoffs/`, `guardrails.py` |
| `crewai/` | Standalone CrewAI projects (`coder`, `debate`, `engineering_team`, `financial_researcher`, `stock_picker`) — **each with its own `pyproject.toml`, `uv.lock` and README** |
| `langgraph/` | `chatbot.py`, `langgraph_agent.py`, `sidekick/` — Gradio assistant with a browser (Playwright) |
| `autogen/` | `autogen_core`, `autogen_distributed` (gRPC), `database_chat` (SQLite `tickets.db`), `tools_chat` |
| `mcp/` | `first_mcp.py` (agent + Playwright/Filesystem MCP), `poker_mcp.py` (MCP server), `trading_app/` (trading simulator) |
| `page_bot_ai/` | Chatbot that crawls a website (PocketFlow + OpenAI) |
| `sandbox/` | The only folder agents are allowed to write files to |

## Installation

Requires Python 3.12 (`.python-version`), [`uv`](https://docs.astral.sh/uv/), Node.js/`npx` (Playwright, Filesystem, Brave Search, Memory MCP servers).

```bash
uv sync                        # shared environment for everything except crewai/*
cp .env.example .env           # fill in the keys
```

`crewai/*` have their own environments, so run them from the project folder, e.g. `cd crewai/debate && crewai run` (CrewAI is pinned to `1.14.3`).

## Environment variables

`.env` is in `.gitignore`, so keys are never committed. The template is `.env.example`.

| Variable | Where it is needed |
|---|---|
| `OPENAI_API_KEY` | Almost everywhere |
| `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY` | `foundation/` (model comparison) |
| `PUSHOVER_USER`, `PUSHOVER_TOKEN` | Push notifications: `foundation/pushover.py`, `langgraph/sidekick`, `mcp/trading_app` |
| `SENDGRID_API_KEY`, `FROM_EMAIL`, `TO_EMAIL`, `REPLY_TO_EMAIL` | Sending email in `openai/` |
| `SERPER_API_KEY` | Web search: `langgraph/sidekick`, CrewAI |
| `LANGSMITH_*` | LangGraph tracing |
| `BRAVE_API_KEY`, `POLYGON_API_KEY` | `mcp/trading_app` (see below) |

## Running

Scripts are run from the repo root with `uv run <path>`, e.g. `uv run openai/guardrails.py` or `uv run mcp/first_mcp.py`. There are two exceptions:

- `page_bot_ai` is a package, run it with `uv run python -m page_bot_ai`.
- `langgraph/sidekick` and `mcp/trading_app` import modules directly, so run them from their own folder (`cd mcp/trading_app && uv run app.py`).

## mcp/trading_app

Four trader agents (Warren, George, Ray, Cathie) trade through their own MCP servers. All configuration lives in `mcp/trading_app/config.py`.

```bash
cd mcp/trading_app
uv run reset.py           # reset accounts to the starting strategies ($10,000)
uv run trading_floor.py   # scheduler: runs the traders every RUN_EVERY_N_MINUTES (60)
uv run app.py             # Gradio dashboard
```

- Required keys: `OPENAI_API_KEY` and `BRAVE_API_KEY`. `POLYGON_API_KEY` is also required unless `MARKET_SIMULATION=true` is set (random prices, demo only).
- `POLYGON_PLAN=paid|realtime` enables the official `mcp_polygon`.
- `USE_MANY_MODELS=true` runs the traders on different models. This additionally requires `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`, `GROK_API_KEY` and `OPENROUTER_API_KEY`.
- The account is bound to the MCP server process via `ACCOUNT_NAME` rather than chosen by the model. This protects against prompt injection.
- The MCP stdio client does not pass environment secrets to child processes, so all keys are passed explicitly via `env` in `mcp_params.py`.
- Test only against an isolated database: `TRADING_DB_PATH=/tmp/test.db MARKET_SIMULATION=true uv run trading_floor.py`.

## Do not commit

`.env`, `mcp/trading_app/accounts.db*`, `mcp/trading_app/memory/*.db*` (already in `.gitignore`).
