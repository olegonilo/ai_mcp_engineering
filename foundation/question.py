import os
import sys
import textwrap

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv(override=True)

MODEL_FAST = "gpt-4.1-nano"
MODEL_DEFAULT = "gpt-4.1-mini"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_api_key() -> None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        sys.exit(
            "OpenAI API Key not set – please add it to your .env file "
            "or set the OPENAI_API_KEY environment variable."
        )
    print(f"OpenAI API Key found: {key[:8]}…")


def _print_section(title: str, content: str) -> None:
    width = 72
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")
    for line in content.splitlines():
        for chunk in textwrap.wrap(line, width) or [""]:
            print(chunk)


def chat(
        client: OpenAI,
        messages: list[dict[str, str]],
        model: str = MODEL_DEFAULT,
) -> str:
    """Return the assistant reply for the given message history."""
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def run_math_sanity_check(client: OpenAI) -> None:
    """Quick smoke-test: trivial arithmetic."""
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    answer = chat(client, messages, model=MODEL_FAST)
    _print_section("Sanity check – 2 + 2 = ?", answer)


def run_iq_challenge(client: OpenAI) -> None:
    """Generate a challenging IQ question then answer it in the same conversation."""
    messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                "Please propose a hard, challenging question to assess someone's IQ. "
                "Respond only with the question."
            ),
        }
    ]
    question = chat(client, messages)
    _print_section("IQ Challenge – Question", question)

    # Keep the question in context so the model can answer its own question.
    messages.append({"role": "assistant", "content": question})
    messages.append({"role": "user", "content": "Now please answer that question thoroughly."})
    answer = chat(client, messages)
    _print_section("IQ Challenge – Answer", answer)


def run_agentic_opportunity_finder(client: OpenAI) -> None:
    """
    Multi-turn conversation that:
      1. Identifies a promising business area for Agentic AI.
      2. Surfaces the most significant pain-point in that area.
      3. Proposes a concrete Agentic AI solution for the pain-point.

    Each turn builds on the previous one so the model retains full context.
    """
    messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                "Help me pick a business area that might be worth exploring "
                "for an Agentic AI opportunity."
            ),
        }
    ]
    business_area = chat(client, messages)
    _print_section("Agentic AI – Business Area", business_area)

    messages.append({"role": "assistant", "content": business_area})
    messages.append(
        {
            "role": "user",
            "content": "What is the most significant pain-point in that industry?",
        }
    )
    pain_point = chat(client, messages)
    _print_section("Agentic AI – Pain Point", pain_point)

    messages.append({"role": "assistant", "content": pain_point})
    messages.append(
        {
            "role": "user",
            "content": (
                "Now propose a concrete Agentic AI solution that addresses that pain-point."
            ),
        }
    )
    solution = chat(client, messages)
    _print_section("Agentic AI – Proposed Solution", solution)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _validate_api_key()
    client = OpenAI()

    try:
        run_math_sanity_check(client)
        run_iq_challenge(client)
        run_agentic_opportunity_finder(client)
    except OpenAIError as exc:
        sys.exit(f"OpenAI API error: {exc}")


if __name__ == "__main__":
    main()
