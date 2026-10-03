from mcp.server.fastmcp import FastMCP

from accounts import Account
from config import require_env

mcp = FastMCP("accounts_server")


def _bound_account() -> Account:
    # The account is bound to this server process by the launcher, not chosen by the model:
    # otherwise a prompt injection (e.g. from a fetched web page) could trade another trader's account.
    return Account.get(require_env("ACCOUNT_NAME"))


@mcp.tool()
def get_balance() -> float:
    """Get the cash balance of your account."""
    return _bound_account().balance


@mcp.tool()
def get_holdings() -> dict[str, int]:
    """Get the holdings of your account as a mapping of symbol to quantity."""
    return _bound_account().holdings


@mcp.tool()
def buy_shares(symbol: str, quantity: int, rationale: str) -> str:
    """Buy shares of a stock.

    Args:
        symbol: The symbol of the stock
        quantity: The quantity of shares to buy (positive whole number)
        rationale: The rationale for the purchase and fit with the account's strategy
    """
    return _bound_account().buy_shares(symbol, quantity, rationale)


@mcp.tool()
def sell_shares(symbol: str, quantity: int, rationale: str) -> str:
    """Sell shares of a stock.

    Args:
        symbol: The symbol of the stock
        quantity: The quantity of shares to sell (positive whole number)
        rationale: The rationale for the sale and fit with the account's strategy
    """
    return _bound_account().sell_shares(symbol, quantity, rationale)


@mcp.tool()
def change_strategy(strategy: str) -> str:
    """At your discretion, if you choose to, call this to change your investment strategy for the future.

    Args:
        strategy: The new strategy for the account
    """
    return _bound_account().change_strategy(strategy)


@mcp.resource("accounts://accounts_server/{name}")
def read_account_resource(name: str) -> str:
    return Account.get(name.lower()).report()


@mcp.resource("accounts://strategy/{name}")
def read_strategy_resource(name: str) -> str:
    return Account.get(name.lower()).get_strategy()


if __name__ == "__main__":
    mcp.run(transport="stdio")
