import concurrent.futures
import json
import os
import sys
from collections.abc import Callable

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv(override=True)

JUDGE_MODEL = "gpt-4o-mini"
QUESTION_MODEL = "gpt-4o-mini"

Messages = list[dict[str, str]]
CompetitorFn = Callable[[Messages], str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_keys() -> None:
    keys = [
        ("OpenAI", os.getenv("OPENAI_API_KEY"), 8, True),
        ("Anthropic", os.getenv("ANTHROPIC_API_KEY"), 7, False),
        ("Google", os.getenv("GOOGLE_API_KEY"), 2, False),
        ("DeepSeek", os.getenv("DEEPSEEK_API_KEY"), 3, False),
        ("Groq", os.getenv("GROQ_API_KEY"), 4, False),
    ]
    for name, key, prefix, required in keys:
        if key:
            print(f"{name} API Key exists and begins {key[:prefix]}")
        elif required:
            sys.exit(f"{name} API Key is required but not set")
        else:
            print(f"{name} API Key not set (optional)")


def call_openai_compat(client: OpenAI, model: str, msgs: Messages) -> str:
    return client.chat.completions.create(model=model, messages=msgs).choices[0].message.content or ""


def call_claude(client: Anthropic, model: str, msgs: Messages) -> str:
    return client.messages.create(model=model, messages=msgs, max_tokens=1000).content[0].text


def query_competitor(name: str, fn: CompetitorFn, msgs: Messages) -> tuple[str, str]:
    try:
        answer = fn(msgs)
        print(f"  [{name}] responded.")
        return name, answer
    except Exception as e:
        print(f"  [{name}] failed: {e}")
        return name, f"[ERROR: {e}]"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def generate_question(openai_client: OpenAI) -> str:
    """Stage 1 — Prompt Chaining: generate a benchmark question."""
    print("\n--- Stage 1: Generating benchmark question ---")
    request = (
        "Please come up with a challenging, nuanced question that I can ask a number of LLMs "
        "to evaluate their intelligence. Answer only with the question, no explanation."
    )
    question = call_openai_compat(openai_client, QUESTION_MODEL, [{"role": "user", "content": request}])
    print(f"Question: {question}\n")
    return question


def collect_answers(
        competitors: list[tuple[str, CompetitorFn]],
        question: str,
) -> list[tuple[str, str]]:
    """Stage 2 — Parallelization (Voting): fan-out to all competitors concurrently."""
    print("--- Stage 2: Collecting competitor responses (parallel) ---")
    msgs: Messages = [{"role": "user", "content": question}]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(query_competitor, name, fn, msgs): name for name, fn in competitors}
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    order = {name: i for i, (name, _) in enumerate(competitors)}
    results.sort(key=lambda r: order[r[0]])

    for name, answer in results:
        print(f"\n=== {name} ===\n{answer}")

    return results


def judge_responses(
        openai_client: OpenAI,
        question: str,
        results: list[tuple[str, str]],
) -> None:
    """Stage 3 — Prompt Chaining: judge ranks all responses."""
    print("\n--- Stage 3: Judging responses ---")
    competitor_names = [name for name, _ in results]
    combined = "\n\n".join(
        f"# Response from competitor {i + 1}\n\n{answer}"
        for i, (_, answer) in enumerate(results)
    )
    judge_prompt = f"""You are judging a competition between {len(competitor_names)} competitors.
Each model has been given this question:

{question}

Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": ["best competitor number", "second best competitor number", ...]}}

Here are the responses from each competitor:

{combined}

Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""

    raw_ranking = call_openai_compat(openai_client, JUDGE_MODEL, [{"role": "user", "content": judge_prompt}])
    ranks = json.loads(raw_ranking)["results"]

    print("\n--- Final Rankings ---")
    for rank, result in enumerate(ranks, start=1):
        print(f"Rank {rank}: {competitor_names[int(result) - 1]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    validate_keys()

    openai_client = OpenAI()
    claude_client = Anthropic()
    gemini_client = OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    deepseek_client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
    groq_client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    competitors: list[tuple[str, CompetitorFn]] = [
        ("gpt-4o-mini", lambda msgs: call_openai_compat(openai_client, "gpt-4o-mini", msgs)),
        ("claude-sonnet-4-5", lambda msgs: call_claude(claude_client, "claude-sonnet-4-5", msgs)),
        ("gemini-2.5-flash", lambda msgs: call_openai_compat(gemini_client, "gemini-2.5-flash", msgs)),
        ("deepseek-chat", lambda msgs: call_openai_compat(deepseek_client, "deepseek-chat", msgs)),
        ("openai/gpt-oss-120b", lambda msgs: call_openai_compat(groq_client, "openai/gpt-oss-120b", msgs)),
        ("llama3.1:8b", lambda msgs: call_openai_compat(ollama_client, "llama3.1:8b", msgs)),
    ]

    try:
        question = generate_question(openai_client)
        results = collect_answers(competitors, question)
        judge_responses(openai_client, question, results)
    except OpenAIError as exc:
        sys.exit(f"OpenAI API error: {exc}")


if __name__ == "__main__":
    main()
