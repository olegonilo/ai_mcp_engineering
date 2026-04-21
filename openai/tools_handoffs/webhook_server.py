"""
SDR Reply Webhook — SendGrid Inbound Parse + OpenAI Agents SDK
==============================================================

Flow
----
1. Your SDR sends a personalized email with Reply-To: replies@parse.yourdomain.com
2. Prospect hits Reply and sends back to that address
3. SendGrid Inbound Parse intercepts it, POSTs multipart/form-data here (/inbound)
4. The SDR Reply Agent reads the prospect's message and sends a contextual reply

One-time setup
--------------
1. Add an MX record for your parse subdomain:
     Hostname : parse.yourdomain.com
     Mail server: mx.sendgrid.net
     Priority  : 10
   (Use a subdomain you control that is NOT your authenticated sending domain)

2. In SendGrid Dashboard → Settings → Inbound Parse → Add Host & URL:
     Subdomain : parse
     Domain    : yourdomain.com
     Destination URL: https://your-server.com/inbound

3. In .env:
     FROM_EMAIL=you@yourdomain.com
     REPLY_TO_EMAIL=replies@parse.yourdomain.com
     SENDGRID_API_KEY=SG.xxx

Local dev (ngrok)
-----------------
   uvicorn webhook_server:app --reload --port 8000
   ngrok http 8000
   # paste the https ngrok URL into SendGrid as the Destination URL
"""

import os
import re
import sys

import sendgrid
import uvicorn
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from sendgrid.helpers.mail import Content, Email, Mail, ReplyTo, To

load_dotenv(override=True)

app = FastAPI()

MODEL = "gpt-4o-mini"
FROM_EMAIL = os.getenv("FROM_EMAIL", "o.onilo@enjoygaming.com")
REPLY_TO_EMAIL = os.getenv("REPLY_TO_EMAIL", "o.onilo@enjoygaming.com")

# Mirror of the prospect list in tools_handoffs.py — used to look up prospect context
PROSPECTS: dict[str, dict] = {
    "sarah@techflow.com": {"name": "Sarah Chen", "company": "TechFlow Inc", "role": "CTO"},
    "marcus@financecore.com": {"name": "Marcus Rodriguez", "company": "FinanceCore Ltd", "role": "VP Engineering"},
    "priya@cloudsync.ai": {"name": "Priya Patel", "company": "CloudSync AI", "role": "Head of Security"},
    "james@datavault.io": {"name": "James O'Brien", "company": "DataVault Systems", "role": "CISO"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sg_client() -> sendgrid.SendGridAPIClient:
    api_key = os.environ.get("SENDGRID_API_KEY")
    if not api_key:
        sys.exit("SENDGRID_API_KEY is required but not set")
    return sendgrid.SendGridAPIClient(api_key=api_key)


def _extract_email(raw: str) -> str:
    """Pull a bare email address out of a 'Name <email>' or plain address string."""
    match = re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", raw)
    return match.group(0).lower() if match else raw.lower()


def _strip_quoted_reply(text: str) -> str:
    """
    Remove the quoted original message from a reply body so the agent only
    sees what the prospect actually wrote this time.

    Handles the most common quoting conventions:
      - Gmail / Outlook  "On Mon, 12 Jan ... wrote:"
      - Inline >-prefixed quoted lines
      - "--- Original Message ---" / "______" separators
    """
    patterns = [
        r"\n\s*On .+?wrote:\s*",  # Gmail / Apple Mail
        r"\n[ \t]*>[ \t]?.*",  # > quoted lines
        r"\n-{3,}\s*Original Message.*",  # Outlook separator
        r"\n_{5,}.*",  # long underscore separator
        r"\nFrom:.*\nSent:.*\nTo:.*",  # Outlook header block
    ]
    for pattern in patterns:
        text = re.split(pattern, text, maxsplit=1, flags=re.DOTALL | re.IGNORECASE)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# Agent factory — one agent per inbound request, tool bound to that prospect
# ---------------------------------------------------------------------------

def _make_sdr_agent(prospect_email: str, prospect_name: str, prospect_company: str) -> Agent:
    """
    Build an SDR Reply Agent with a send_reply tool pre-bound to the prospect.
    Defining the tool inside the factory lets us close over the prospect's
    email address without passing it as an agent parameter.
    """

    @function_tool
    def send_reply(html_body: str) -> dict:
        """Send the SDR's reply to the prospect as an HTML email."""
        subject = f"Re: Streamlining SOC2 compliance at {prospect_company}"
        mail = Mail(Email(FROM_EMAIL), To(prospect_email), subject, Content("text/html", html_body))
        mail.reply_to = ReplyTo(REPLY_TO_EMAIL)  # keep the reply chain alive
        _sg_client().client.mail.send.post(request_body=mail.get())
        print(f"  [SDR] Reply sent → {prospect_name} <{prospect_email}>")
        return {"status": "sent", "to": prospect_email}

    @function_tool
    def book_demo(proposed_times: str) -> dict:
        """
        Signal that the prospect is interested in a demo.
        proposed_times: comma-separated list of suggested time slots.
        In production wire this to Calendly / HubSpot / your CRM.
        """
        print(f"  [SDR] Demo interest logged for {prospect_name}. Proposed: {proposed_times}")
        return {"status": "demo_requested", "prospect": prospect_email, "times": proposed_times}

    return Agent(
        name="SDR Reply Agent",
        model=MODEL,
        tools=[send_reply, book_demo],
        instructions=f"""You are Alex, an SDR at ComplAI — a SaaS platform that automates SOC2 \
compliance and audit preparation using AI. You are in an active email conversation with \
{prospect_name}, {prospect_company}.

Your goal: keep the conversation moving toward a 20-minute product demo.

Guidelines:
- Be warm and genuinely helpful. Never sound scripted or pushy.
- Interest / curiosity → acknowledge, answer briefly, invite them to book a demo \
  (use the book_demo tool to log their interest and propose 2-3 time slots).
- Questions about the product → answer concisely, then offer to show them live.
- Objections (cost, timing, priority) → empathize, address the concern in one sentence, \
  keep the door open with a soft ask.
- Not interested → thank them graciously; wish them well. Do not push further.
- Keep replies to 3-5 short sentences. No walls of text. No bullet lists unless asked.
- Write clean HTML: wrap paragraphs in <p> tags. No heavy inline styles.
- Sign every reply: <p>— Alex | ComplAI</p>

Use send_reply once to send your response. Use book_demo only when the prospect \
signals genuine interest in seeing the product.""",
    )


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------

@app.post("/inbound")
async def inbound_email(request: Request):
    """
    Receive a prospect reply from SendGrid Inbound Parse.
    SendGrid POSTs multipart/form-data; we must return 2xx or it will retry.
    """
    form = await request.form()

    raw_from: str = form.get("from", "")
    subject: str = form.get("subject", "No subject")
    # Prefer plain text; fall back to HTML (agent handles either)
    raw_body: str = form.get("text") or form.get("html") or ""

    sender_email = _extract_email(raw_from)
    body = _strip_quoted_reply(raw_body)

    if not body:
        print(f"[Inbound] Ignored empty reply from {sender_email}")
        return {"status": "ignored", "reason": "empty body after stripping quoted text"}

    print(f"\n[Inbound] From   : {raw_from}")
    print(f"[Inbound] Subject: {subject}")
    print(f"[Inbound] Body   : {body[:300]}{'...' if len(body) > 300 else ''}")

    # Look up prospect — fall back gracefully for unknown senders
    prospect = PROSPECTS.get(sender_email, {
        "name": sender_email.split("@")[0].capitalize(),
        "company": sender_email.split("@")[1] if "@" in sender_email else "Unknown",
        "role": "contact",
    })

    agent = _make_sdr_agent(sender_email, prospect["name"], prospect["company"])

    prompt = (
        f"Email reply from {prospect['name']} "
        f"({prospect['role']} at {prospect['company']}):\n\n"
        f"Subject: {subject}\n\n"
        f"{body}"
    )

    result = await Runner.run(agent, prompt)
    print(f"[SDR Agent] Done. Final output: {result.final_output}")

    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("webhook_server:app", host="0.0.0.0", port=8000, reload=True)
