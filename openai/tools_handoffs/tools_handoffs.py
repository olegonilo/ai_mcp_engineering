import asyncio
import os
import sys
from typing import Dict, List

import sendgrid
from agents import Agent, Runner, trace, function_tool
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from sendgrid.helpers.mail import Content, Email, Mail, ReplyTo, To

load_dotenv(override=True)

MODEL = "gpt-4o-mini"
FROM_EMAIL = os.getenv("FROM_EMAIL")
if not FROM_EMAIL:
    sys.exit("FROM_EMAIL is required but not set")
TO_EMAIL = os.getenv("TO_EMAIL")
if not TO_EMAIL:
    sys.exit("TO_EMAIL is required but not set")

# Replies from prospects land here — must be your SendGrid Inbound Parse subdomain
REPLY_TO_EMAIL = os.getenv("REPLY_TO_EMAIL")
if not REPLY_TO_EMAIL:
    sys.exit("REPLY_TO_EMAIL is required but not set")

# Mock prospect list — replace with a CSV loader or CRM integration in production
PROSPECTS: list[dict] = [
    {"name": "Sarah Chen", "company": "TechFlow Inc", "role": "CTO", "email": "sarah@techflow.com"},
    {"name": "Marcus Rodriguez", "company": "FinanceCore Ltd", "role": "VP Engineering",
     "email": "marcus@financecore.com"},
    {"name": "Priya Patel", "company": "CloudSync AI", "role": "Head of Security", "email": "priya@cloudsync.ai"},
    {"name": "James O'Brien", "company": "DataVault Systems", "role": "CISO", "email": "james@datavault.io"},
]

# In-memory campaign log — tracks every personalized send this session
_campaign_log: list[dict] = []


# ---------------------------------------------------------------------------
# Email tools
# ---------------------------------------------------------------------------

def _sg_client() -> sendgrid.SendGridAPIClient:
    api_key = os.environ.get("SENDGRID_API_KEY")
    if not api_key:
        sys.exit("SENDGRID_API_KEY is required but not set")
    return sendgrid.SendGridAPIClient(api_key=api_key)


@function_tool
def send_email(body: str) -> Dict[str, str]:
    """Send a plain-text email to the default recipient."""
    mail = Mail(Email(FROM_EMAIL), To(TO_EMAIL), "Sales email", Content("text/plain", body)).get()
    _sg_client().client.mail.send.post(request_body=mail)
    return {"status": "success"}


@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """Send an HTML email with the given subject to the default recipient."""
    mail = Mail(Email(FROM_EMAIL), To(TO_EMAIL), subject, Content("text/html", html_body)).get()
    _sg_client().client.mail.send.post(request_body=mail)
    return {"status": "success"}


@function_tool
def get_prospect_list() -> List[dict]:
    """Return the list of sales prospects for the mail merge campaign."""
    return PROSPECTS


@function_tool
def send_personalized_email(
        recipient_name: str,
        recipient_email: str,
        subject: str,
        html_body: str,
) -> Dict[str, str]:
    """Send a personalized HTML email to a single prospect and log the result."""
    mail = Mail(Email(FROM_EMAIL), To(recipient_email), subject, Content("text/html", html_body))
    mail.reply_to = ReplyTo(REPLY_TO_EMAIL)  # prospect replies route to SendGrid Inbound Parse
    _sg_client().client.mail.send.post(request_body=mail.get())
    entry = {"recipient": recipient_name, "email": recipient_email, "subject": subject, "status": "sent"}
    _campaign_log.append(entry)
    print(f"  [Campaign] Sent → {recipient_name} <{recipient_email}> | {subject}")
    return entry


@function_tool
def get_campaign_report() -> Dict:
    """Return a summary of all emails sent in the current campaign session."""
    return {"total_sent": len(_campaign_log), "recipients": _campaign_log}


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def make_sales_agents() -> tuple[Agent, Agent, Agent]:
    base = "You work for ComplAI, a SaaS platform for SOC2 compliance and audit preparation, powered by AI."
    return (
        Agent(name="Professional Sales Agent", model=MODEL,
              instructions=f"You are a professional, serious sales agent. {base} Write formal cold emails."),
        Agent(name="Engaging Sales Agent", model=MODEL,
              instructions=f"You are a witty, engaging sales agent. {base} Write humorous cold emails likely to get a response."),
        Agent(name="Concise Sales Agent", model=MODEL,
              instructions=f"You are a busy, direct sales agent. {base} Write concise, to-the-point cold emails."),
    )


def _make_support_agents() -> tuple[Agent, Agent, Agent]:
    """Return (subject_writer, html_converter, personalizer) — shared across stages."""
    subject_writer = Agent(
        name="Email Subject Writer",
        model=MODEL,
        instructions="Write a compelling subject line for a cold sales email. Reply with only the subject line.",
    )
    html_converter = Agent(
        name="HTML Email Converter",
        model=MODEL,
        instructions=(
            "Convert a plain-text or markdown email body to a clean, compelling HTML email. "
            "Reply with only the HTML."
        ),
    )
    personalizer = Agent(
        name="Email Personalizer",
        model=MODEL,
        instructions=(
            "Personalize a cold sales email for a specific prospect. "
            "Given a base email and prospect details (name, company, role), rewrite it to address them personally — "
            "reference their name, company, and role naturally. Keep the core message intact. "
            "Reply with only the personalized email body."
        ),
    )
    return subject_writer, html_converter, personalizer


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

async def stream_single_email(agent: Agent, message: str) -> None:
    """Stage 1: Stream one cold email to stdout."""
    print("\n--- Stage 1: Streaming cold email ---")
    result = Runner.run_streamed(agent, input=message)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
    print()


async def run_parallel_drafts(agents: tuple[Agent, Agent, Agent], message: str) -> None:
    """Stage 2: Generate three email drafts in parallel, then pick the best."""
    print("\n--- Stage 2: Parallel drafts + selection ---")
    with trace("Parallel cold emails"):
        results = await asyncio.gather(
            Runner.run(agents[0], message),
            Runner.run(agents[1], message),
            Runner.run(agents[2], message),
        )
    outputs = [r.final_output for r in results]
    for i, output in enumerate(outputs, 1):
        print(f"\n--- Draft {i} ---\n{output}")

    picker = Agent(
        name="Sales Picker",
        model=MODEL,
        instructions=(
            "You pick the best cold sales email from the given options. "
            "Imagine you are a customer and pick the one you are most likely to respond to. "
            "Do not give an explanation; reply with the selected email only."
        ),
    )
    emails = "Cold sales emails:\n\n" + "\n\nEmail:\n\n".join(outputs)
    with trace("Email selection"):
        best = await Runner.run(picker, emails)
    print(f"\nBest email selected:\n{best.final_output}")


async def run_sales_manager(agents: tuple[Agent, Agent, Agent], message: str) -> None:
    """Stage 3: Sales Manager orchestrates drafts and sends the best via send_email tool."""
    print("\n--- Stage 3: Sales Manager — orchestrate and send ---")
    tool1 = agents[0].as_tool(tool_name="sales_agent1", tool_description="Write a professional cold sales email")
    tool2 = agents[1].as_tool(tool_name="sales_agent2", tool_description="Write a witty, engaging cold sales email")
    tool3 = agents[2].as_tool(tool_name="sales_agent3", tool_description="Write a concise cold sales email")

    manager = Agent(
        name="Sales Manager",
        model=MODEL,
        tools=[tool1, tool2, tool3, send_email],
        instructions=(
            "You are a Sales Manager at ComplAI. Find the single best cold sales email.\n\n"
            "Steps:\n"
            "1. Use all three sales_agent tools to generate three drafts.\n"
            "2. Pick the single most effective email.\n"
            "3. Send it using the send_email tool — exactly once.\n\n"
            "Rules: never write drafts yourself; send exactly one email."
        ),
    )
    with trace("Sales manager send"):
        await Runner.run(manager, message)


async def run_automated_sdr(agents: tuple[Agent, Agent, Agent], message: str) -> None:
    """Stage 4: Sales Manager picks the best draft, hands off to Email Manager for HTML formatting and send."""
    print("\n--- Stage 4: Automated SDR — handoff to Email Manager ---")
    tool1 = agents[0].as_tool(tool_name="sales_agent1", tool_description="Write a professional cold sales email")
    tool2 = agents[1].as_tool(tool_name="sales_agent2", tool_description="Write a witty, engaging cold sales email")
    tool3 = agents[2].as_tool(tool_name="sales_agent3", tool_description="Write a concise cold sales email")

    subject_writer, html_converter, _ = _make_support_agents()

    emailer_agent = Agent(
        name="Email Manager",
        model=MODEL,
        handoff_description="Format an email as HTML and send it to a single recipient",
        tools=[
            subject_writer.as_tool(tool_name="subject_writer",
                                   tool_description="Write a subject for a cold sales email"),
            html_converter.as_tool(tool_name="html_converter", tool_description="Convert email body to HTML"),
            send_html_email,
        ],
        instructions=(
            "You receive an email body. "
            "Use subject_writer to craft a subject, html_converter to convert the body to HTML, "
            "then send it with send_html_email."
        ),
    )

    sales_manager = Agent(
        name="Sales Manager",
        model=MODEL,
        tools=[tool1, tool2, tool3],
        handoffs=[emailer_agent],
        instructions=(
            "You are a Sales Manager at ComplAI. Find the single best cold sales email.\n\n"
            "Steps:\n"
            "1. Use all three sales_agent tools to generate three drafts.\n"
            "2. Pick the single most effective email.\n"
            "3. Hand off only the winning draft to the Email Manager for formatting and sending.\n\n"
            "Rules: never write drafts yourself; hand off exactly one email."
        ),
    )
    with trace("Automated SDR"):
        await Runner.run(sales_manager, message)


async def run_mail_merge_campaign(agents: tuple[Agent, Agent, Agent], message: str) -> None:
    """Stage 5: Mail Merge Agent drafts, personalizes, and sends HTML emails to every prospect."""
    print("\n--- Stage 5: Mail Merge Campaign ---")
    _campaign_log.clear()

    tool1 = agents[0].as_tool(tool_name="sales_agent1", tool_description="Write a professional cold sales email")
    tool2 = agents[1].as_tool(tool_name="sales_agent2", tool_description="Write a witty, engaging cold sales email")
    tool3 = agents[2].as_tool(tool_name="sales_agent3", tool_description="Write a concise cold sales email")

    subject_writer, html_converter, personalizer = _make_support_agents()

    mail_merge_agent = Agent(
        name="Mail Merge Agent",
        model=MODEL,
        tools=[
            tool1,
            tool2,
            tool3,
            get_prospect_list,
            personalizer.as_tool(
                tool_name="personalize_email",
                tool_description="Personalize a base email for a specific prospect given their name, company, and role",
            ),
            subject_writer.as_tool(
                tool_name="subject_writer",
                tool_description="Write a compelling subject line for a cold sales email",
            ),
            html_converter.as_tool(
                tool_name="html_converter",
                tool_description="Convert a plain-text or markdown email body to HTML",
            ),
            send_personalized_email,
            get_campaign_report,
        ],
        instructions=(
            "You are a Mail Merge Specialist at ComplAI running a personalized outreach campaign.\n\n"
            "Steps:\n"
            "1. Use all three sales_agent tools to generate three different cold email drafts.\n"
            "2. Pick the single strongest draft as your base template.\n"
            "3. Use get_prospect_list to retrieve all prospects.\n"
            "4. For each prospect:\n"
            "   a. Call personalize_email with the base template and the prospect's name, company, and role.\n"
            "   b. Call subject_writer to write a subject line that references their role or company.\n"
            "   c. Call html_converter to convert the personalized body to HTML.\n"
            "   d. Call send_personalized_email with their name, email, subject, and HTML body.\n"
            "5. After all prospects are emailed, call get_campaign_report and summarize the results.\n\n"
            "Rules: every email must be individually personalized; never send the same copy to everyone."
        ),
    )

    with trace("Mail merge campaign"):
        result = await Runner.run(mail_merge_agent, message)

    print(f"\nCampaign complete:\n{result.final_output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    agents = make_sales_agents()

    await stream_single_email(agents[0], "Write a cold sales email")
    await run_parallel_drafts(agents, "Write a cold sales email")
    await run_sales_manager(agents, "Send a cold sales email addressed to 'Dear CEO'")
    await run_automated_sdr(agents, "Send out a cold sales email addressed to Dear CEO from Alice")
    await run_mail_merge_campaign(agents, "Run a personalized email campaign to our prospect list")


if __name__ == "__main__":
    asyncio.run(main())
