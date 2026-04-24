import os
import sys

import sendgrid
from sendgrid.helpers.mail import Content, Email, Mail, To
from agents import Agent, function_tool

FROM_EMAIL = os.getenv("FROM_EMAIL")
if not FROM_EMAIL:
    sys.exit("FROM_EMAIL is required but not set")

TO_EMAIL = os.getenv("TO_EMAIL")
if not TO_EMAIL:
    sys.exit("TO_EMAIL is required but not set")


@function_tool
def send_email(subject: str, html_body: str) -> dict[str, str]:
    """Send an email with the given subject and HTML body."""
    api_key = os.environ.get("SENDGRID_API_KEY")
    if not api_key:
        raise RuntimeError("SENDGRID_API_KEY is required but not set")
    sg = sendgrid.SendGridAPIClient(api_key=api_key)
    mail = Mail(Email(FROM_EMAIL), To(TO_EMAIL), subject, Content("text/html", html_body)).get()
    response = sg.client.mail.send.post(request_body=mail)
    print(f"Email response: {response.status_code}")
    return {"status": "success"}


INSTRUCTIONS = (
    "You are able to send a nicely formatted HTML email based on a detailed report. "
    "You will be provided with a detailed report. You should use your tool to send one email, "
    "providing the report converted into clean, well presented HTML with an appropriate subject line."
)

email_agent = Agent(
    name="Email agent",
    instructions=INSTRUCTIONS,
    tools=[send_email],
    model="gpt-4o-mini",
)
