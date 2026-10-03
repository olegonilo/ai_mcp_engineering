import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel

from config import INITIAL_BALANCE, MAX_TIME_SERIES_POINTS, REPORT_TRANSACTIONS, SPREAD
from database import connection, read_account, write_account, write_log
from market import PriceUnavailableError, get_share_price, normalize_symbol

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _validate_quantity(quantity: int) -> None:
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError(f"Quantity must be a positive whole number, got {quantity!r}")


class Transaction(BaseModel):
    symbol: str
    quantity: int
    price: float
    timestamp: str
    rationale: str

    def total(self) -> float:
        return self.quantity * self.price

    def __repr__(self):
        return f"{abs(self.quantity)} shares of {self.symbol} at {self.price} each."


class Account(BaseModel):
    name: str
    balance: float
    strategy: str
    holdings: dict[str, int]
    transactions: list[Transaction]
    portfolio_value_time_series: list[tuple[str, float]]
    # Cash added via deposit() minus cash removed via withdraw(); needed for a correct P&L.
    net_deposits: float = 0.0

    @staticmethod
    def _new_fields(name: str) -> dict:
        return {
            "name": name.lower(),
            "balance": INITIAL_BALANCE,
            "strategy": "",
            "holdings": {},
            "transactions": [],
            "portfolio_value_time_series": [],
        }

    @classmethod
    def get(cls, name: str) -> "Account":
        fields = read_account(name)
        if fields:
            return cls(**fields)
        with connection(write=True) as conn:
            fields = read_account(name, conn)
            if not fields:
                fields = cls._new_fields(name)
                write_account(name, fields, conn)
        return cls(**fields)

    def _apply(self, mutate: Callable[["Account"], T]) -> T:
        """Re-read, mutate and persist the account inside one write transaction, so concurrent
        writers (several MCP server processes, the scheduler) can never overwrite each other."""
        with connection(write=True) as conn:
            fields = read_account(self.name, conn) or self._new_fields(self.name)
            fresh = Account(**fields)
            result = mutate(fresh)
            write_account(fresh.name, fresh.model_dump(), conn)
        for field in type(self).model_fields:
            setattr(self, field, getattr(fresh, field))
        return result

    def save(self) -> None:
        write_account(self.name, self.model_dump())

    def reset(self, strategy: str) -> None:
        def mutate(acc: "Account") -> None:
            acc.balance = INITIAL_BALANCE
            acc.strategy = strategy
            acc.holdings = {}
            acc.transactions = []
            acc.portfolio_value_time_series = []
            acc.net_deposits = 0.0

        self._apply(mutate)

    def deposit(self, amount: float) -> None:
        """Deposit funds into the account."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive.")

        def mutate(acc: "Account") -> None:
            acc.balance = round(acc.balance + amount, 2)
            acc.net_deposits = round(acc.net_deposits + amount, 2)

        self._apply(mutate)
        logger.info("Deposited $%s into %s. New balance: $%s", amount, self.name, self.balance)

    def withdraw(self, amount: float) -> None:
        """Withdraw funds from the account, ensuring it doesn't go negative."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive.")

        def mutate(acc: "Account") -> None:
            if amount > acc.balance:
                raise ValueError("Insufficient funds for withdrawal.")
            acc.balance = round(acc.balance - amount, 2)
            acc.net_deposits = round(acc.net_deposits - amount, 2)

        self._apply(mutate)
        logger.info("Withdrew $%s from %s. New balance: $%s", amount, self.name, self.balance)

    def buy_shares(self, symbol: str, quantity: int, rationale: str) -> str:
        """Buy shares of a stock if sufficient funds are available."""
        _validate_quantity(quantity)
        symbol = normalize_symbol(symbol)
        # Price lookup is a network call: do it before taking the database write lock.
        buy_price = round(get_share_price(symbol) * (1 + SPREAD), 4)
        total_cost = round(buy_price * quantity, 2)

        def mutate(acc: "Account") -> None:
            if total_cost > acc.balance:
                raise ValueError(
                    f"Insufficient funds: {quantity} {symbol} costs ${total_cost:,.2f}, "
                    f"balance is ${acc.balance:,.2f}."
                )
            acc.holdings[symbol] = acc.holdings.get(symbol, 0) + quantity
            acc.transactions.append(
                Transaction(symbol=symbol, quantity=quantity, price=buy_price, timestamp=_now(), rationale=rationale)
            )
            acc.balance = round(acc.balance - total_cost, 2)

        self._apply(mutate)
        write_log(self.name, "account", f"Bought {quantity} of {symbol}")
        return self._completed_message(f"Bought {quantity} {symbol} at ${buy_price:,.4f}.")

    def sell_shares(self, symbol: str, quantity: int, rationale: str) -> str:
        """Sell shares of a stock if the user has enough shares."""
        _validate_quantity(quantity)
        symbol = normalize_symbol(symbol)
        if self.holdings.get(symbol, 0) < quantity:
            raise ValueError(f"Cannot sell {quantity} shares of {symbol}. Not enough shares held.")
        sell_price = round(get_share_price(symbol) * (1 - SPREAD), 4)
        total_proceeds = round(sell_price * quantity, 2)

        def mutate(acc: "Account") -> None:
            held = acc.holdings.get(symbol, 0)
            if held < quantity:
                raise ValueError(f"Cannot sell {quantity} shares of {symbol}. Not enough shares held.")
            if held == quantity:
                del acc.holdings[symbol]
            else:
                acc.holdings[symbol] = held - quantity
            # Negative quantity marks a sale.
            acc.transactions.append(
                Transaction(symbol=symbol, quantity=-quantity, price=sell_price, timestamp=_now(), rationale=rationale)
            )
            acc.balance = round(acc.balance + total_proceeds, 2)

        self._apply(mutate)
        write_log(self.name, "account", f"Sold {quantity} of {symbol}")
        return self._completed_message(f"Sold {quantity} {symbol} at ${sell_price:,.4f}.")

    def _completed_message(self, summary: str) -> str:
        # The trade is already committed; report() never raises on valuation failure, so the
        # model cannot mistake a successful trade for an error and place a duplicate order.
        return f"Completed. {summary} Latest details:\n" + self.report(record=True)

    def calculate_portfolio_value(self) -> float:
        """Calculate the total value of the user's portfolio."""
        return self.balance + sum(get_share_price(symbol) * qty for symbol, qty in self.holdings.items())

    def calculate_profit_loss(self, portfolio_value: float | None = None) -> float:
        """Profit or loss relative to the initial balance plus any net deposits."""
        if portfolio_value is None:
            portfolio_value = self.calculate_portfolio_value()
        return portfolio_value - INITIAL_BALANCE - self.net_deposits

    def record_portfolio_value(self) -> float:
        """Append the current portfolio value to the time series and persist it."""
        value = round(self.calculate_portfolio_value(), 2)

        def mutate(acc: "Account") -> None:
            acc.portfolio_value_time_series.append((_now(), value))
            del acc.portfolio_value_time_series[:-MAX_TIME_SERIES_POINTS]

        self._apply(mutate)
        return value

    def get_holdings(self) -> dict[str, int]:
        """Report the current holdings of the user."""
        return self.holdings

    def get_profit_loss(self) -> float:
        """Report the user's profit or loss at any point in time."""
        return self.calculate_profit_loss()

    def list_transactions(self) -> list[dict]:
        """List all transactions made by the user."""
        return [transaction.model_dump() for transaction in self.transactions]

    def report(self, record: bool = False) -> str:
        """Return a compact JSON string describing the account (sent to the LLM).

        The time series and old transactions are omitted: they grow without bound and would
        inflate every prompt. With record=True the valuation is also appended to the time series.
        """
        data = self.model_dump(exclude={"portfolio_value_time_series", "transactions"})
        data["recent_transactions"] = self.list_transactions()[-REPORT_TRANSACTIONS:]
        try:
            portfolio_value = self.record_portfolio_value() if record else self.calculate_portfolio_value()
        except PriceUnavailableError as e:
            logger.warning("Valuation unavailable for %s: %s", self.name, e)
            data["valuation_error"] = str(e)
        else:
            data["total_portfolio_value"] = round(portfolio_value, 2)
            data["total_profit_loss"] = round(self.calculate_profit_loss(portfolio_value), 2)
        write_log(self.name, "account", "Retrieved account details")
        return json.dumps(data)

    def get_strategy(self) -> str:
        """Return the strategy of the account"""
        write_log(self.name, "account", "Retrieved strategy")
        return self.strategy

    def change_strategy(self, strategy: str) -> str:
        """At your discretion, if you choose to, call this to change your investment strategy for the future"""
        if not strategy.strip():
            raise ValueError("Strategy must not be empty.")

        def mutate(acc: "Account") -> None:
            acc.strategy = strategy

        self._apply(mutate)
        write_log(self.name, "account", "Changed strategy")
        return "Changed strategy"


if __name__ == "__main__":
    account = Account.get("John Doe")
    account.deposit(1000)
    print(account.buy_shares("AAPL", 5, "example"))
    print(account.sell_shares("AAPL", 2, "example"))
    print(f"Current Holdings: {account.get_holdings()}")
    print(f"Total Portfolio Value: {account.calculate_portfolio_value()}")
    print(f"Profit/Loss: {account.get_profit_loss()}")
    print(f"Transactions: {account.list_transactions()}")
