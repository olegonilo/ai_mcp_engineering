from __future__ import annotations

import os

import openai
from dotenv import load_dotenv

from .config import LLM_MODEL, LLM_TEMPERATURE

_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        load_dotenv(override=True)
        _client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def call_llm(prompt: str) -> str:
    """Send a user prompt to the LLM and return the text response."""
    response = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
    )
    return response.choices[0].message.content
