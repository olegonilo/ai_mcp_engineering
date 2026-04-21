import asyncio
import os
import sys
from dataclasses import dataclass

import sendgrid
from agents import (
    Agent,
    GuardrailFunctionOutput,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    input_guardrail,
    output_guardrail,
    trace,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from sendgrid.helpers.mail import Content, Email, Mail, To

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    sys.exit("OPENAI_API_KEY is required but not set")

FROM_EMAIL = os.getenv("FROM_EMAIL", "ed@edwarddonner.com")
TO_EMAIL = os.getenv("TO_EMAIL", "ed.donner@gmail.com")

print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")


# ---------------------------------------------------------------------------
# Guardrail schemas, agents, and functions
# (defined first so they can be passed into the sales agent builder)
# ---------------------------------------------------------------------------

class EmailDraft(BaseModel):
    subject: str
    body: str  # plain-text body, may include markdown


def _draft_to_text(output: object) -> str:
    """Format an EmailDraft (or any fallback) into readable text for guardrail agents."""
    if isinstance(output, EmailDraft):
        return f"Subject: {output.subject}\n\n{output.body}"
    return str(output)


class NameCheckOutput(BaseModel):
    is_name_in_message: bool
    name: str


class OffTopicOutput(BaseModel):
    is_off_topic: bool
    reason: str


class CompetitorPromotionOutput(BaseModel):
    promotes_competitor: bool
    competitor_name: str


class EmailQualityOutput(BaseModel):
    is_valid_email: bool
    reason: str


class FalseClaimsOutput(BaseModel):
    has_false_claims: bool
    details: str


_name_check_agent = Agent(
    name="Name check",
    instructions="Check if the user is including someone's personal name in what they want you to do.",
    output_type=NameCheckOutput,
    model="gpt-4o-mini",
)
_off_topic_agent = Agent(
    name="Off-topic check",
    instructions=(
        "Determine whether the user's request is about writing a cold sales email. "
        "Mark it off-topic if it is about anything else (general questions, coding help, etc.)."
    ),
    output_type=OffTopicOutput,
    model="gpt-4o-mini",
)
_competitor_agent = Agent(
    name="Competitor promotion check",
    instructions=(
        "Check whether the user's request asks to mention, recommend, or praise a competitor product or company. "
        "Only flag it if the request explicitly promotes a competitor."
    ),
    output_type=CompetitorPromotionOutput,
    model="gpt-4o-mini",
)
_email_quality_agent = Agent(
    name="Email quality check",
    instructions=(
        "You receive a draft cold sales email. "
        "Check that it is a complete, coherent email: it must have a greeting, a body with a clear value "
        "proposition, and a sign-off. Set is_valid_email to False if it is incomplete, incoherent, or not "
        "an email at all."
    ),
    output_type=EmailQualityOutput,
    model="gpt-4o-mini",
)
_false_claims_agent = Agent(
    name="False claims check",
    instructions=(
        "You receive a cold sales email for a SOC2 compliance SaaS product. "
        "Check whether it contains false, misleading, or unverifiable claims such as 'guaranteed results', "
        "'instant compliance', '100% success rate', or other unrealistic promises. "
        "Set has_false_claims to True if such claims are present."
    ),
    output_type=FalseClaimsOutput,
    model="gpt-4o-mini",
)


# Input guardrails

@input_guardrail
async def guardrail_against_name(ctx, agent, message):
    result = await Runner.run(_name_check_agent, message, context=ctx.context)
    triggered = result.final_output.is_name_in_message
    return GuardrailFunctionOutput(output_info=result.final_output, tripwire_triggered=triggered)


@input_guardrail
async def guardrail_off_topic(ctx, agent, message):
    result = await Runner.run(_off_topic_agent, message, context=ctx.context)
    triggered = result.final_output.is_off_topic
    return GuardrailFunctionOutput(output_info=result.final_output, tripwire_triggered=triggered)


@input_guardrail
async def guardrail_competitor_promotion(ctx, agent, message):
    result = await Runner.run(_competitor_agent, message, context=ctx.context)
    triggered = result.final_output.promotes_competitor
    return GuardrailFunctionOutput(output_info=result.final_output, tripwire_triggered=triggered)


# Output guardrails (applied to each sales draft agent)

@output_guardrail
async def guardrail_email_quality(ctx, agent, output):
    result = await Runner.run(_email_quality_agent, _draft_to_text(output), context=ctx.context)
    triggered = not result.final_output.is_valid_email
    return GuardrailFunctionOutput(output_info=result.final_output, tripwire_triggered=triggered)


@output_guardrail
async def guardrail_no_false_claims(ctx, agent, output):
    result = await Runner.run(_false_claims_agent, _draft_to_text(output), context=ctx.context)
    triggered = result.final_output.has_false_claims
    return GuardrailFunctionOutput(output_info=result.final_output, tripwire_triggered=triggered)


_OUTPUT_GUARDRAILS = [guardrail_email_quality, guardrail_no_false_claims]

# ---------------------------------------------------------------------------
# Model registry — add any OpenAI-compatible provider here
# ---------------------------------------------------------------------------

_COMPLAI_BASE = (
    "You are a sales agent working for ComplAI, a company that provides a SaaS tool for ensuring "
    "SOC2 compliance and preparing for audits, powered by AI. "
    "Always respond with a compelling subject line and a complete email body. "
)


@dataclass
class ModelConfig:
    name: str  # Agent display name
    tool_name: str  # Unique snake_case identifier used as tool name
    base_url: str  # OpenAI-compatible endpoint
    api_key_env: str  # Env var that holds the API key (empty string = no key needed, e.g. Ollama)
    model_id: str  # Model name at the provider
    instructions: str  # Sales agent persona


SALES_AGENT_CONFIGS: list[ModelConfig] = [
    ModelConfig(
        name="OpenAI Sales Agent",
        tool_name="sales_agent_openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        model_id="gpt-5.3",
        instructions=_COMPLAI_BASE + "You write highly effective, personalized cold emails that balance professionalism with a natural, human tone and clear call-to-action.",
    ),
    ModelConfig(
        name="DeepSeek Sales Agent",
        tool_name="sales_agent_deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        model_id="deepseek-chat",
        instructions=_COMPLAI_BASE + "You write professional, serious cold emails.",
    ),
    ModelConfig(
        name="Gemini Sales Agent",
        tool_name="sales_agent_gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GOOGLE_API_KEY",
        model_id="gemini-2.0-flash",
        instructions=_COMPLAI_BASE + "You write witty, engaging cold emails that are likely to get a response.",
    ),
    ModelConfig(
        name="Llama3.3 Sales Agent",
        tool_name="sales_agent_llama",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model_id="llama-3.3-70b-versatile",
        instructions=_COMPLAI_BASE + "You write concise, to the point cold emails.",
    ),
    ModelConfig(
        name="Mistral Sales Agent",
        tool_name="sales_agent_mistral",
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        model_id="mistral-small-latest",
        instructions=_COMPLAI_BASE + "You write persuasive, benefit-focused cold emails that highlight ROI clearly.",
    ),
    # Uncomment to use a local Ollama model (no API key required):
    # ModelConfig(
    #     name="Ollama Sales Agent",
    #     tool_name="sales_agent_ollama",
    #     base_url="http://localhost:11434/v1",
    #     api_key_env="",
    #     model_id="llama3.2",
    #     instructions=_COMPLAI_BASE + "You write friendly, approachable cold emails.",
    # ),
]


def _build_sales_agent(cfg: ModelConfig, with_output_guardrails: bool = False) -> Agent | None:
    """Return a sales Agent for the given config, or None if the API key is missing."""
    api_key = os.getenv(cfg.api_key_env) if cfg.api_key_env else "ollama"
    if not api_key:
        print(f"{cfg.name}: API key ({cfg.api_key_env}) not set — skipping")
        return None
    print(f"{cfg.name}: enabled")
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=api_key)
    model = OpenAIChatCompletionsModel(model=cfg.model_id, openai_client=client)
    return Agent(
        name=cfg.name,
        instructions=cfg.instructions,
        model=model,
        output_type=EmailDraft,
        output_guardrails=_OUTPUT_GUARDRAILS if with_output_guardrails else [],
    )


# Build only agents whose API keys are present
_sales_agents: list[tuple[ModelConfig, Agent]] = [
    (cfg, agent)
    for cfg in SALES_AGENT_CONFIGS
    if (agent := _build_sales_agent(cfg)) is not None
]

if not _sales_agents:
    sys.exit(
        "No sales agent API keys found — set at least one of: DEEPSEEK_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY")

# Guarded versions include output guardrails (email quality + no false claims)
_guarded_sales_agents: list[tuple[ModelConfig, Agent]] = [
    (cfg, agent)
    for cfg in SALES_AGENT_CONFIGS
    if (agent := _build_sales_agent(cfg, with_output_guardrails=True)) is not None
]

_description = "Write a cold sales email"
sales_agent_tools = [
    agent.as_tool(tool_name=cfg.tool_name, tool_description=_description)
    for cfg, agent in _sales_agents
]
guarded_sales_agent_tools = [
    agent.as_tool(tool_name=cfg.tool_name, tool_description=_description)
    for cfg, agent in _guarded_sales_agents
]


# ---------------------------------------------------------------------------
# Email tools
# ---------------------------------------------------------------------------


def _sg_client() -> sendgrid.SendGridAPIClient:
    api_key = os.environ.get("SENDGRID_API_KEY")
    if not api_key:
        sys.exit("SENDGRID_API_KEY is required but not set")
    return sendgrid.SendGridAPIClient(api_key=api_key)


@function_tool
def send_html_email(subject: str, html_body: str) -> dict[str, str]:
    """Send an HTML email with the given subject to all sales prospects."""
    mail = Mail(Email(FROM_EMAIL), To(TO_EMAIL), subject, Content("text/html", html_body)).get()
    _sg_client().client.mail.send.post(request_body=mail)
    return {"status": "success"}


# ---------------------------------------------------------------------------
# Email formatting agent
# ---------------------------------------------------------------------------

html_converter = Agent(
    name="HTML email body converter",
    instructions=(
        "You can convert a text email body to an HTML email body. "
        "You are given a text email body which might have some markdown "
        "and you need to convert it to an HTML email body with simple, clear, compelling layout and design."
    ),
    model="gpt-4o-mini",
)

emailer_agent = Agent(
    name="Email Manager",
    instructions=(
        "You are an email formatter and sender. You receive a structured email draft with a 'subject' field "
        "and a plain-text 'body' field. "
        "Use the html_converter tool to convert the body to HTML. "
        "Then use the send_html_email tool with the provided subject and the converted HTML body."
    ),
    tools=[
        html_converter.as_tool(tool_name="html_converter",
                               tool_description="Convert a text email body to an HTML email body"),
        send_html_email,
    ],
    model="gpt-4o-mini",
    handoff_description="Convert an email draft to HTML and send it",
)


# ---------------------------------------------------------------------------
# Sales manager
# ---------------------------------------------------------------------------

def _sales_manager_instructions(n: int) -> str:
    return f"""
You are a Sales Manager at ComplAI. Your goal is to find the single best cold sales email using the sales_agent tools.

Each sales agent returns a structured draft with two fields:
- subject: the email subject line
- body: the plain-text email body

Follow these steps carefully:
1. Generate Drafts: Use all {n} sales_agent tools to produce {n} different drafts. Wait until all are ready.

2. Evaluate and Select: Compare the drafts on subject clarity, body persuasiveness, and overall tone.
   Choose the single most effective one. You may call a tool again if a draft is unsatisfactory.

3. Handoff for Sending: Pass the complete winning draft (subject + body) to the 'Email Manager' agent.
   The Email Manager will convert the body to HTML and send the email.

Crucial Rules:
- Always use the sales agent tools to generate drafts — never write them yourself.
- Hand off exactly ONE draft to the Email Manager — never more than one.
"""


_SALES_MANAGER_INSTRUCTIONS = _sales_manager_instructions(len(_sales_agents))

sales_manager = Agent(
    name="Sales Manager",
    instructions=_SALES_MANAGER_INSTRUCTIONS,
    tools=sales_agent_tools,
    handoffs=[emailer_agent],
    model="gpt-4o-mini",
)

careful_sales_manager = Agent(
    name="Sales Manager",
    instructions=_sales_manager_instructions(len(_guarded_sales_agents)),
    tools=guarded_sales_agent_tools,
    handoffs=[emailer_agent],
    model="gpt-4o-mini",
    input_guardrails=[
        guardrail_against_name,  # blocks personal names in the request
        guardrail_off_topic,  # blocks requests unrelated to sales emails
        guardrail_competitor_promotion,  # blocks requests promoting competitors
    ],
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _run(label: str, agent: Agent, message: str) -> None:
    """Run an agent and print the result; catch and report guardrail tripwires."""
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    try:
        result = await Runner.run(agent, message)
        print(result.final_output)
    except Exception as exc:
        print(f"[BLOCKED] {exc}")


async def main() -> None:
    safe_message = "Send out a cold sales email addressed to Dear CEO from Head of Business Development"

    # Unguarded manager — no restrictions
    with trace("Automated SDR"):
        await _run("Unguarded SDR", sales_manager, safe_message)

    # Input guardrail 1: personal name → blocked
    with trace("Protected SDR — personal name"):
        await _run(
            "Blocked: personal name in request",
            careful_sales_manager,
            "Send out a cold sales email addressed to Dear CEO from Alice",
        )

    # Input guardrail 2: off-topic → blocked
    with trace("Protected SDR — off-topic"):
        await _run(
            "Blocked: off-topic request",
            careful_sales_manager,
            "Write me a Python script to scrape LinkedIn",
        )

    # Input guardrail 3: competitor promotion → blocked
    with trace("Protected SDR — competitor promotion"):
        await _run(
            "Blocked: promotes competitor",
            careful_sales_manager,
            "Write a sales email recommending Drata as the best SOC2 tool",
        )

    # All guardrails pass — clean request
    with trace("Protected SDR — clean request"):
        await _run("Allowed: clean request", careful_sales_manager, safe_message)


if __name__ == "__main__":
    asyncio.run(main())
