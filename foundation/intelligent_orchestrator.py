import concurrent.futures
import json
import os
import sys
from collections.abc import Callable

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QUESTION_MODEL = "gpt-4o-mini"
ORCHESTRATOR_MODEL = "gpt-4o-mini"
SYNTHESIZER_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-4o-mini"

Messages = list[dict[str, str]]
ModelFn = Callable[[Messages], str]


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
    return client.messages.create(model=model, messages=msgs, max_tokens=800).content[0].text


def parse_json(raw: str) -> dict:
    """Parse JSON, stripping any markdown code fences the model may include."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def generate_question(openai_client: OpenAI) -> str:
    """Stage 1: Generate a challenging benchmark question."""
    print("\n--- Stage 1: Generating benchmark question ---")
    request = (
        "Please come up with a challenging, nuanced question that I can ask a number of LLMs "
        "to evaluate their intelligence. Answer only with the question, no explanation."
    )
    question = call_openai_compat(openai_client, QUESTION_MODEL, [{"role": "user", "content": request}])
    print(f"Question: {question}\n")
    return question


def orchestrate(openai_client: OpenAI, question: str, model_names: list[str]) -> list[dict]:
    """Stage 2: Decompose the question and route each sub-question to the best model."""
    print("--- Stage 2: Orchestrating — decomposing question and routing ---")

    model_descriptions = {
        "gpt-4o-mini": "Reasoning, complex logic, nuanced analysis",
        "claude-sonnet-4-5": "Creative writing, empathy, ethical reasoning",
        "gemini-2.5-flash": "Factual retrieval, technical explanations, structured data",
        "deepseek-chat": "Code generation, mathematical problems, technical documentation",
        "openai/gpt-oss-120b": "General purpose, cost-effective for straightforward tasks",
        "llama3.2": "Privacy-focused, sensitive data, general tasks",
    }
    model_list = "\n".join(
        f"- {name}: {model_descriptions.get(name, 'General purpose')}"
        for name in model_names
    )
    prompt = f"""You are an intelligent orchestrator AI. Analyze this complex question and:

1. Break it down into 3-4 simpler sub-questions
2. For each sub-question, recommend which model is best suited

Available models and their strengths:
{model_list}

Original question: {question}

Respond with JSON only, in this format:
{{
    "sub_questions": [
        {{
            "question": "the sub-question text",
            "reasoning": "why this model is best for this sub-question",
            "recommended_model": "model_name"
        }}
    ]
}}"""

    raw = call_openai_compat(openai_client, ORCHESTRATOR_MODEL, [{"role": "user", "content": prompt}])
    sub_questions = parse_json(raw)["sub_questions"]

    print("Routing plan:")
    for i, item in enumerate(sub_questions, 1):
        print(f"  {i}. [{item['recommended_model']}] {item['question']}")
        print(f"     Reason: {item['reasoning']}")
    return sub_questions


def _run_sub_question(
        idx: int,
        item: dict,
        model_fns: dict[str, ModelFn],
        fallback_fn: ModelFn,
) -> tuple[str, dict]:
    """Execute one sub-question, falling back to the default model on error."""
    sub_q = item["question"]
    model = item["recommended_model"]
    msgs: Messages = [{"role": "user", "content": sub_q}]
    fn = model_fns.get(model, fallback_fn)

    try:
        answer = fn(msgs)
        print(f"  [{idx}] {model} — done")
    except Exception as exc:
        print(f"  [{idx}] {model} failed ({exc}), using fallback")
        answer = fallback_fn(msgs)
        model = f"{FALLBACK_MODEL} (fallback)"

    return sub_q, {"model": model, "answer": answer, "reasoning": item["reasoning"]}


def execute_sub_questions(
        sub_questions: list[dict],
        model_fns: dict[str, ModelFn],
        fallback_fn: ModelFn,
) -> dict[str, dict]:
    """Stage 3: Execute all sub-questions in parallel, preserving submission order."""
    print("\n--- Stage 3: Executing sub-questions (parallel) ---")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_run_sub_question, idx, item, model_fns, fallback_fn)
            for idx, item in enumerate(sub_questions, 1)
        ]
        results = dict(f.result() for f in futures)

    for sub_q, data in results.items():
        print(f"\n=== Sub-question ===\n{sub_q}")
        print(f"Model: {data['model']}\n{data['answer']}")

    return results


def synthesize(openai_client: OpenAI, question: str, sub_answers: dict[str, dict]) -> str:
    """Stage 4: Combine all specialized answers into one coherent response."""
    print("\n--- Stage 4: Synthesizing final answer ---")
    sep = "=" * 60
    sections = f"\n{sep}\n".join(
        f"SUB-QUESTION: {sub_q}\nASSIGNED TO: {data['model']}\n"
        f"REASONING: {data['reasoning']}\nANSWER: {data['answer']}"
        for sub_q, data in sub_answers.items()
    )
    prompt = (
        "You are a synthesis AI combining specialized responses into a comprehensive answer.\n\n"
        f"ORIGINAL QUESTION: {question}\n\n"
        f"{sep}\n{sections}\n{sep}\n\n"
        "Synthesize these responses into one coherent, comprehensive answer. "
        "Highlight how different model strengths contributed to the final answer."
    )
    answer = call_openai_compat(openai_client, SYNTHESIZER_MODEL, [{"role": "user", "content": prompt}])
    print(f"\n{'=' * 72}\nFINAL ANSWER\n{'=' * 72}\n{answer}")
    return answer


def print_analysis(sub_answers: dict[str, dict]) -> None:
    """Print a summary of the agentic patterns used in this run."""
    model_lines = "\n".join(
        f"  - {data['model']}: {data['reasoning']}" for data in sub_answers.values()
    )
    total = len(sub_answers) + 3  # question + orchestrate + workers + synthesize
    print(f"""
{'=' * 72}
PATTERN ANALYSIS
{'=' * 72}

Patterns (Anthropic's Building Effective Agents):
  1. Orchestrator-Workers  — one LLM decomposes and routes tasks to specialists
  2. Intelligent Routing   — each sub-task matched to the best available model
  3. Parallelization       — workers run concurrently via ThreadPoolExecutor

Models used:
{model_lines}

API calls: 1 (question) + 1 (orchestration) + {len(sub_answers)} (workers) + 1 (synthesis) = {total} total
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    validate_keys()

    openai_client = OpenAI()
    claude_client = Anthropic()
    gemini_client = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"),
                           base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
    groq_client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    model_fns: dict[str, ModelFn] = {
        "gpt-4o-mini": lambda msgs: call_openai_compat(openai_client, "gpt-4o-mini", msgs),
        "claude-sonnet-4-5": lambda msgs: call_claude(claude_client, "claude-sonnet-4-5", msgs),
        "gemini-2.5-flash": lambda msgs: call_openai_compat(gemini_client, "gemini-2.5-flash", msgs),
        "deepseek-chat": lambda msgs: call_openai_compat(deepseek_client, "deepseek-chat", msgs),
        "openai/gpt-oss-120b": lambda msgs: call_openai_compat(groq_client, "openai/gpt-oss-120b", msgs),
        "llama3.2": lambda msgs: call_openai_compat(ollama_client, "llama3.2", msgs),
    }
    fallback_fn: ModelFn = lambda msgs: call_openai_compat(openai_client, FALLBACK_MODEL, msgs)

    try:
        question = generate_question(openai_client)
        sub_questions = orchestrate(openai_client, question, list(model_fns.keys()))
        sub_answers = execute_sub_questions(sub_questions, model_fns, fallback_fn)
        synthesize(openai_client, question, sub_answers)
        print_analysis(sub_answers)
    except OpenAIError as exc:
        sys.exit(f"OpenAI API error: {exc}")


if __name__ == "__main__":
    main()
