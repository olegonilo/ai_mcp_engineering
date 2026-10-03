"""Launch parameters for the MCP servers used by traders and researchers.

The MCP stdio client only passes a small allow-list of variables (PATH, HOME, ...) to child
processes, so every secret a server needs must be forwarded explicitly via "env".
"""

import os

from config import BASE_DIR, MEMORY_DIR, POLYGON_API_KEY, POLYGON_PLAN, is_paid_polygon, is_realtime_polygon, require_env


def _local_server(script: str, env: dict[str, str] | None = None) -> dict:
    return {
        "command": "uv",
        "args": ["run", str(BASE_DIR / script)],
        "cwd": str(BASE_DIR),
        "env": {k: v for k, v in (env or {}).items() if v is not None},
    }


def _market_env() -> dict[str, str | None]:
    return {
        "POLYGON_API_KEY": POLYGON_API_KEY,
        "POLYGON_PLAN": POLYGON_PLAN,
        "MARKET_SIMULATION": os.getenv("MARKET_SIMULATION"),
        "TRADING_DB_PATH": os.getenv("TRADING_DB_PATH"),
    }


def market_mcp() -> dict:
    """The MCP server for the Trader to read Market Data."""
    if is_paid_polygon or is_realtime_polygon:
        return {
            "command": "uvx",
            # TODO(prod): pin to an immutable commit SHA instead of a (mutable) tag.
            "args": ["--from", "git+https://github.com/polygon-io/mcp_polygon@v0.1.0", "mcp_polygon"],
            "env": {"POLYGON_API_KEY": require_env("POLYGON_API_KEY")},
        }
    return _local_server("market_server.py", _market_env())


def trader_mcp_server_params(name: str) -> list[dict]:
    """The MCP servers for one trader: Accounts (bound to this trader), Push Notification and the Market."""
    return [
        _local_server("accounts_server.py", {**_market_env(), "ACCOUNT_NAME": name.lower()}),
        _local_server(
            "push_server.py",
            {"PUSHOVER_USER": os.getenv("PUSHOVER_USER"), "PUSHOVER_TOKEN": os.getenv("PUSHOVER_TOKEN")},
        ),
        market_mcp(),
    ]


def researcher_mcp_server_params(name: str) -> list[dict]:
    """The MCP servers for the researcher: Fetch, Brave Search and Memory."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return [
        {"command": "uvx", "args": ["mcp-server-fetch"]},
        {
            "command": "npx",
            # TODO(prod): pin npm package versions; "-y" without a version runs whatever is latest.
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": require_env("BRAVE_API_KEY")},
        },
        {
            "command": "npx",
            "args": ["-y", "mcp-memory-libsql"],
            "env": {"LIBSQL_URL": f"file:{MEMORY_DIR / f'{name.lower()}.db'}"},
        },
    ]
