from __future__ import annotations

from urllib.parse import urlparse

from pocketflow import Flow

from .agent import AgentDecision, DraftAnswer
from .crawler import CrawlAndExtract


def create_flow() -> Flow:
    """Build the agentic crawl → decide → answer flow."""
    crawl = CrawlAndExtract()
    decide = AgentDecision()
    draft = DraftAnswer()

    crawl >> decide
    decide - "explore" >> crawl
    decide - "answer" >> draft

    return Flow(start=crawl)


def run_chatbot(
        question: str,
        target_urls: list[str],
        instruction: str = "Provide helpful and accurate answers.",
) -> str:
    """Crawl *target_urls* and answer *question* using an agentic loop.

    Args:
        question:    The user's question.
        target_urls: Seed URLs; only pages within their domains are crawled.
        instruction: Behavioural guidance injected into every LLM prompt.

    Returns:
        A Markdown-formatted answer string.
    """
    print(f"\n{'=' * 60}")
    print(f"Question:    {question}")
    print(f"Target URLs: {target_urls}")
    print(f"Instruction: {instruction}")
    print(f"{'=' * 60}\n")

    allowed_domains = [urlparse(url).netloc for url in target_urls]
    shared: dict = {
        "user_question": question,
        "instruction": instruction,
        "allowed_domains": allowed_domains,
        "all_discovered_urls": list(target_urls),
        "visited_urls": set(),
        "url_content": {},
        "urls_to_process": list(range(len(target_urls))),
        "current_iteration": 0,
        "final_answer": None,
    }

    create_flow().run(shared)
    return shared.get("final_answer") or "No answer generated."
