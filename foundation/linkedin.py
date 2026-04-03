import json
import os
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent
ME_DIR = ROOT / "me"
LINKEDIN_PDF = ME_DIR / "linkedin.pdf"
SUMMARY_PATH = ME_DIR / "summary.txt"

NAME = "Oleh Onilo"
CHAT_MODEL = "gpt-4o-mini"
EVALUATOR_MODEL = "gpt-4.1-nano"


class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str


def _openai_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Add it to your .env or environment.")
    return OpenAI()


def load_linkedin_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text.strip())
    return "\n\n".join(parts)


def context_block(summary: str, linkedin: str) -> str:
    return f"## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n"


def build_system_prompt(name: str, summary: str, linkedin: str) -> str:
    ctx = context_block(summary, linkedin)
    return (
        f"You are acting as {name}. You are answering questions on {name}'s website, "
        f"particularly questions related to {name}'s career, background, skills and experience. "
        f"Your responsibility is to represent {name} for interactions on the website as faithfully "
        "as possible. "
        "You are given a summary of their background and LinkedIn profile which you can use to "
        "answer questions. "
        "Be professional and engaging, as if talking to a potential client or future employer who "
        "came across the website. "
        "If you don't know the answer, say so.\n\n"
        f"{ctx}\n"
        f"With this context, please chat with the user, always staying in character as {name}."
    )


def build_evaluator_system_prompt(name: str, summary: str, linkedin: str) -> str:
    ctx = context_block(summary, linkedin)
    return (
        "You are an evaluator that decides whether a response to a question is acceptable. "
        "You are provided with a conversation between a User and an Agent. "
        "Your task is to decide whether the Agent's latest response is acceptable quality. "
        f"The Agent is playing the role of {name} and is representing {name} on their website. "
        "The Agent has been instructed to be professional and engaging, as if talking to a "
        "potential client or future employer who came across the website. "
        f"The Agent has been provided with context on {name} in the form of their summary and "
        "LinkedIn details. Here's the information:\n\n"
        f"{ctx}\n"
        "With this context, please evaluate the latest response, replying with whether the "
        "response is acceptable and your feedback."
    )


def _history_as_text(history: list) -> str:
    try:
        return json.dumps(history, ensure_ascii=False, indent=2)
    except TypeError:
        return repr(history)


def evaluator_user_prompt(reply: str, message: str, history: list) -> str:
    return (
        f"Here's the conversation between the User and the Agent:\n\n{_history_as_text(history)}\n\n"
        f"Here's the latest message from the User:\n\n{message}\n\n"
        f"Here's the latest response from the Agent:\n\n{reply}\n\n"
        "Please evaluate the response, replying with whether it is acceptable and your feedback."
    )


def chat_completion(client: OpenAI, messages: list, model: str) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return (response.choices[0].message.content or "").strip()


def evaluate(
        client: OpenAI,
        evaluator_system: str,
        reply: str,
        message: str,
        history: list,
) -> Evaluation:
    messages = [
        {"role": "system", "content": evaluator_system},
        {"role": "user", "content": evaluator_user_prompt(reply, message, history)},
    ]
    response = client.beta.chat.completions.parse(
        model=EVALUATOR_MODEL,
        messages=messages,
        response_format=Evaluation,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        return Evaluation(is_acceptable=True, feedback="Structured parse failed; accepting reply.")
    return parsed


def rerun(
        client: OpenAI,
        base_system: str,
        reply: str,
        message: str,
        history: list,
        feedback: str,
) -> str:
    updated = (
        f"{base_system}\n\n## Previous answer rejected\n"
        "You just tried to reply, but the quality control rejected your reply.\n"
        f"## Your attempted answer:\n{reply}\n\n"
        f"## Reason for rejection:\n{feedback}\n\n"
    )
    messages = (
            [{"role": "system", "content": updated}]
            + history
            + [{"role": "user", "content": message}]
    )
    return chat_completion(client, messages, CHAT_MODEL)


def chat(
        message: str,
        history: list,
        *,
        client: OpenAI,
        system_prompt: str,
        evaluator_system: str,
) -> str:
    messages = (
            [{"role": "system", "content": system_prompt}]
            + history
            + [{"role": "user", "content": message}]
    )
    reply = chat_completion(client, messages, CHAT_MODEL)
    evaluation = evaluate(client, evaluator_system, reply, message, history)
    if evaluation.is_acceptable:
        return reply
    return rerun(client, system_prompt, reply, message, history, evaluation.feedback)


def main() -> None:
    if not LINKEDIN_PDF.is_file():
        sys.exit(f"LinkedIn PDF not found: {LINKEDIN_PDF}")
    if not SUMMARY_PATH.is_file():
        sys.exit(f"Summary file not found: {SUMMARY_PATH}")

    linkedin = load_linkedin_text(LINKEDIN_PDF)
    summary = SUMMARY_PATH.read_text(encoding="utf-8")

    client = _openai_client()
    system_prompt = build_system_prompt(NAME, summary, linkedin)
    evaluator_system = build_evaluator_system_prompt(NAME, summary, linkedin)

    def _chat(message: str, history: list) -> str:
        return chat(
            message,
            history,
            client=client,
            system_prompt=system_prompt,
            evaluator_system=evaluator_system,
        )

    try:
        gr.ChatInterface(_chat, type="messages").launch()
    except OpenAIError as exc:
        sys.exit(f"OpenAI-compatible API error: {exc}")


if __name__ == "__main__":
    main()
