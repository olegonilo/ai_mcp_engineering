# AI MCP Engineering

Навчальний монорепозиторій з агентних AI-фреймворків: OpenAI Agents SDK, CrewAI, LangGraph, AutoGen, MCP.

## Структура

| Папка | Що всередині |
|---|---|
| `foundation/` | Базові патерни на «чистих» LLM API: prompt chaining/parallelization, оркестратор, особистий агент з Pushover |
| `openai/` | OpenAI Agents SDK: `deep_research/` (багатоагентне дослідження + email), `tools_handoffs/`, `guardrails.py` |
| `crewai/` | Окремі CrewAI-проєкти (`coder`, `debate`, `engineering_team`, `financial_researcher`, `stock_picker`) — **кожен зі своїм `pyproject.toml`, `uv.lock` і README** |
| `langgraph/` | `chatbot.py`, `langgraph_agent.py`, `sidekick/` — Gradio-асистент з браузером (Playwright) |
| `autogen/` | `autogen_core`, `autogen_distributed` (gRPC), `database_chat` (SQLite `tickets.db`), `tools_chat` |
| `mcp/` | `first_mcp.py` (агент + Playwright/Filesystem MCP), `poker_mcp.py` (MCP-сервер), `trading_app/` (торговий симулятор) |
| `page_bot_ai/` | Чат-бот, що краулить сайт (PocketFlow + OpenAI) |
| `sandbox/` | Єдина папка, куди агентам дозволено писати файли |

## Встановлення

Потрібні Python 3.12 (`.python-version`), [`uv`](https://docs.astral.sh/uv/), Node.js/`npx` (MCP-сервери Playwright, Filesystem, Brave Search, Memory).

```bash
uv sync                        # спільне оточення для всього, крім crewai/*
cp .env.example .env           # заповнити ключі
```

`crewai/*` мають власні оточення, тому запускати їх треба з папки проєкту, наприклад `cd crewai/debate && crewai run` (CrewAI зафіксовано на версії `1.14.3`).

## Змінні оточення

`.env` у `.gitignore`, тому ключі не комітяться. Шаблон лежить у `.env.example`.

| Змінна | Де потрібна |
|---|---|
| `OPENAI_API_KEY` | Майже скрізь |
| `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY` | `foundation/` (порівняння моделей) |
| `PUSHOVER_USER`, `PUSHOVER_TOKEN` | Push-сповіщення: `foundation/pushover.py`, `langgraph/sidekick`, `mcp/trading_app` |
| `SENDGRID_API_KEY`, `FROM_EMAIL`, `TO_EMAIL`, `REPLY_TO_EMAIL` | Відправка email в `openai/` |
| `SERPER_API_KEY` | Веб-пошук: `langgraph/sidekick`, CrewAI |
| `LANGSMITH_*` | Трасування LangGraph |
| `BRAVE_API_KEY`, `POLYGON_API_KEY` | `mcp/trading_app` (див. нижче) |

## Запуск

Скрипти запускаються з кореня командою `uv run <шлях>`, наприклад `uv run openai/guardrails.py` або `uv run mcp/first_mcp.py`. Є два винятки:

- `page_bot_ai` — це пакет, запуск через `uv run python -m page_bot_ai`.
- `langgraph/sidekick` і `mcp/trading_app` імпортують модулі напряму, тож запускати їх треба з власної папки (`cd mcp/trading_app && uv run app.py`).

## mcp/trading_app

Чотири агенти-трейдери (Warren, George, Ray, Cathie) торгують через власні MCP-сервери. Уся конфігурація зібрана в `mcp/trading_app/config.py`.

```bash
cd mcp/trading_app
uv run reset.py           # скинути рахунки до стартових стратегій ($10,000)
uv run trading_floor.py   # планувальник: запуск трейдерів кожні RUN_EVERY_N_MINUTES (60)
uv run app.py             # Gradio-дашборд
```

- Обов'язкові ключі: `OPENAI_API_KEY` і `BRAVE_API_KEY`. `POLYGON_API_KEY` теж потрібен, якщо не ввімкнено `MARKET_SIMULATION=true` (випадкові ціни, лише для демо).
- `POLYGON_PLAN=paid|realtime` вмикає офіційний `mcp_polygon`.
- `USE_MANY_MODELS=true` запускає трейдерів на різних моделях. Для цього додатково потрібні `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`, `GROK_API_KEY` і `OPENROUTER_API_KEY`.
- Рахунок закріплюється за процесом MCP-сервера через `ACCOUNT_NAME`, а не обирається моделлю. Це захист від prompt injection.
- stdio-клієнт MCP не передає дочірнім процесам секрети з оточення, тому всі ключі прокидаються явно через `env` у `mcp_params.py`.
- Тестувати можна тільки на ізольованій базі: `TRADING_DB_PATH=/tmp/test.db MARKET_SIMULATION=true uv run trading_floor.py`.

## Не комітити

`.env`, `mcp/trading_app/accounts.db*`, `mcp/trading_app/memory/*.db*` (уже в `.gitignore`).
