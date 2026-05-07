"""Memory configuration for StockPicker.

Three logical tiers over a single LanceDB store:
  - short_term  (/stock_picker/session)   – in-run context, fades quickly
  - long_term   (/stock_picker/history)   – cross-run decisions, slow decay
  - user        (/stock_picker/user)      – user preferences, always relevant
"""

from pathlib import Path

from crewai.memory import Memory
from crewai.memory.memory_scope import MemoryScope
from crewai.memory.storage.lancedb_storage import LanceDBStorage

_MEMORY_DIR = Path(__file__).parent.parent.parent / "memory"
_KNOWLEDGE_DIR = Path(__file__).parent.parent.parent / "knowledge"

_EMBEDDER = {
    "provider": "openai",
    "config": {"model": "text-embedding-3-small"},
}


def build_memory() -> Memory:
    """Shared LanceDB-backed Memory for the whole crew."""
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return Memory(
        storage=LanceDBStorage(path=_MEMORY_DIR),
        embedder=_EMBEDDER,
        root_scope="/stock_picker",
        recency_half_life_days=30,
        recency_weight=0.3,
        semantic_weight=0.5,
        importance_weight=0.2,
    )


def short_term(mem: Memory) -> MemoryScope:
    """In-run context: high recency, short scope."""
    return MemoryScope(memory=mem, root_path="/stock_picker/session")


def long_term(mem: Memory) -> MemoryScope:
    """Cross-run history: past picks and research decisions."""
    return MemoryScope(memory=mem, root_path="/stock_picker/history")


def user(mem: Memory) -> MemoryScope:
    """User preferences, loaded once from knowledge file."""
    return MemoryScope(memory=mem, root_path="/stock_picker/user")


def seed_user_memory(mem: Memory) -> None:
    """Load user_preference.txt into user memory (idempotent via consolidation)."""
    prefs_file = _KNOWLEDGE_DIR / "user_preference.txt"
    if not prefs_file.exists():
        return
    lines = [ln.strip() for ln in prefs_file.read_text().splitlines() if ln.strip()]
    if not lines:
        return
    scope = user(mem)
    scope.remember_many(
        contents=lines,
        categories=["user_preference"],
        importance=0.9,
        source="knowledge/user_preference.txt",
    )
