import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage

from services.exceptions import AgentExecutionError, DocumentNotFoundError


logger = logging.getLogger(__name__)


class AgentRequestService:
    def __init__(self, agent: Any, user_files: Mapping[str, str]):
        self._agent = agent
        self._user_files = user_files

    async def run(self, user_input: str, user_id: str) -> str:
        if user_id not in self._user_files:
            raise DocumentNotFoundError("No document uploaded for this user_id")

        try:
            new_state = await self._agent.ainvoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "text": self._user_files[user_id],
                    "user_id": user_id,
                }
            )
            return str(new_state["messages"][-1].content)
        except Exception as exc:
            logger.exception("Agent error for user %s", user_id)
            raise AgentExecutionError("Error while using LLM-agent") from exc
