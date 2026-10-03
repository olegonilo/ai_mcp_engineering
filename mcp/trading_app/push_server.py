import logging
import os

import requests
from mcp.server.fastmcp import FastMCP

import config  # noqa: F401  (loads .env and logging)

logger = logging.getLogger(__name__)

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"
PUSH_TIMEOUT_SECONDS = 10
MAX_MESSAGE_LENGTH = 1024  # Pushover limit

mcp = FastMCP("push_server")


@mcp.tool()
def push(message: str) -> str:
    """Send a push notification with this brief message.

    Args:
        message: A brief message to push
    """
    if not pushover_user or not pushover_token:
        raise RuntimeError("Push notifications are not configured (PUSHOVER_USER / PUSHOVER_TOKEN missing)")
    payload = {"user": pushover_user, "token": pushover_token, "message": message[:MAX_MESSAGE_LENGTH]}
    response = requests.post(pushover_url, data=payload, timeout=PUSH_TIMEOUT_SECONDS)
    response.raise_for_status()
    logger.info("Push sent: %s", message[:80])
    return "Push notification sent"


if __name__ == "__main__":
    mcp.run(transport="stdio")
