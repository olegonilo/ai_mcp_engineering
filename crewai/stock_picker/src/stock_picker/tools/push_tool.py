import json
import logging
import os
from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


class PushNotification(BaseModel):
    message: str = Field(..., description="The message to be sent to the user.")


class PushNotificationTool(BaseTool):
    name: str = "Send a Push Notification"
    description: str = "Sends a push notification to the user via Pushover."
    args_schema: Type[BaseModel] = PushNotification

    def _run(self, message: str) -> str:
        user = os.getenv("PUSHOVER_USER")
        token = os.getenv("PUSHOVER_TOKEN")

        if not user or not token:
            return json.dumps({"notification": "error", "reason": "PUSHOVER_USER or PUSHOVER_TOKEN not set"})

        logger.debug("Push: %s", message)
        try:
            response = requests.post(
                _PUSHOVER_URL,
                data={"user": user, "token": token, "message": message},
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            return json.dumps({"notification": "error", "reason": str(e)})

        return json.dumps({"notification": "ok"})
