import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from config import DB_PATH

_initialized = False


def _connect() -> sqlite3.Connection:
    # isolation_level=None: we control transactions explicitly (BEGIN IMMEDIATE below).
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _init_db() -> None:
    global _initialized
    if _initialized:
        return
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    try:
        # WAL lets the UI read while trader processes write.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS accounts (name TEXT PRIMARY KEY, account TEXT)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                datetime DATETIME,
                type TEXT,
                message TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_name_id ON logs (name, id)")
        conn.execute("CREATE TABLE IF NOT EXISTS market (date TEXT PRIMARY KEY, data TEXT)")
    finally:
        conn.close()
    _initialized = True


@contextmanager
def connection(write: bool = False) -> Iterator[sqlite3.Connection]:
    """Yield a connection; with write=True the block runs in one IMMEDIATE transaction,
    which takes the write lock up front and prevents lost updates between processes."""
    _init_db()
    conn = _connect()
    try:
        if write:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
        if write:
            conn.execute("COMMIT")
    except BaseException:
        if write and conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def write_account(name: str, account_dict: dict, conn: sqlite3.Connection | None = None) -> None:
    sql = """
        INSERT INTO accounts (name, account) VALUES (?, ?)
        ON CONFLICT(name) DO UPDATE SET account=excluded.account
    """
    params = (name.lower(), json.dumps(account_dict))
    if conn is not None:
        conn.execute(sql, params)
        return
    with connection(write=True) as own:
        own.execute(sql, params)


def read_account(name: str, conn: sqlite3.Connection | None = None) -> dict | None:
    sql = "SELECT account FROM accounts WHERE name = ?"
    if conn is not None:
        row = conn.execute(sql, (name.lower(),)).fetchone()
    else:
        with connection() as own:
            row = own.execute(sql, (name.lower(),)).fetchone()
    return json.loads(row[0]) if row else None


def write_log(name: str, log_type: str, message: str) -> None:
    """Write a log entry to the logs table."""
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    with connection(write=True) as conn:
        conn.execute(
            "INSERT INTO logs (name, datetime, type, message) VALUES (?, ?, ?, ?)",
            (name.lower(), now, log_type, message),
        )


def read_log(name: str, last_n: int = 10) -> list[tuple[str, str, str]]:
    """Return the most recent (datetime, type, message) entries for a name, oldest first."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT datetime, type, message FROM logs
            WHERE name = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (name.lower(), last_n),
        ).fetchall()
    return rows[::-1]


def write_market(date: str, data: dict) -> None:
    with connection(write=True) as conn:
        conn.execute(
            """
            INSERT INTO market (date, data) VALUES (?, ?)
            ON CONFLICT(date) DO UPDATE SET data=excluded.data
            """,
            (date, json.dumps(data)),
        )


def read_market(date: str) -> dict | None:
    with connection() as conn:
        row = conn.execute("SELECT data FROM market WHERE date = ?", (date,)).fetchone()
    return json.loads(row[0]) if row else None
