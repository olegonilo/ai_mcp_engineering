from __future__ import annotations

from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pocketflow import BatchNode

from .config import CONTENT_MAX_CHARS, MAX_LINKS_PER_PAGE, _CRAWL_HEADERS


def is_valid_url(url: str, allowed_domains: list[str]) -> bool:
    """Return True if *url* belongs to one of the *allowed_domains*."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    domain = parsed.netloc.lower().split(":")[0]
    return any(
        domain == d.lower() or domain.endswith("." + d.lower())
        for d in allowed_domains
    )


def filter_valid_urls(urls: list[str], allowed_domains: list[str]) -> list[str]:
    return [u for u in urls if is_valid_url(u, allowed_domains)]


class CrawlAndExtract(BatchNode):
    """Batch-crawls URLs, extracts clean text, and discovers new links."""

    def prep(self, shared: dict) -> list[tuple[int, str]]:
        discovered = shared.get("all_discovered_urls", [])
        return [
            (idx, discovered[idx])
            for idx in shared.get("urls_to_process", [])
            if idx < len(discovered)
        ]

    def exec(self, url_data: tuple[int, str]) -> tuple[int, str, list[str]]:
        url_idx, url = url_data
        response = requests.get(url, headers=_CRAWL_HEADERS, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        links = [
            urljoin(url, a["href"])
            for a in soup.find_all("a", href=True)
            if urljoin(url, a["href"]).startswith(("http://", "https://"))
        ]
        return url_idx, text, links

    def exec_fallback(self, url_data: tuple[int, str], exc: Exception) -> None:
        _, url = url_data
        print(f"  ✗ Failed to crawl {url}: {type(exc).__name__}: {exc}")
        return None

    def post(self, shared: dict, prep_res, exec_res_list: list) -> None:
        results = [r for r in exec_res_list if r is not None]
        print(f"🔍 Crawled {len(results)} URLs successfully")

        for url_idx, content, links in results:
            truncated = content[:CONTENT_MAX_CHARS]
            if len(content) > CONTENT_MAX_CHARS:
                truncated += "\n... [Content truncated]"

            shared["url_content"][url_idx] = truncated
            shared["visited_urls"].add(url_idx)

            new_links = filter_valid_urls(links, shared["allowed_domains"])[:MAX_LINKS_PER_PAGE]
            for link in new_links:
                if link not in shared["all_discovered_urls"]:
                    shared["all_discovered_urls"].append(link)

        shared["urls_to_process"] = []
