import asyncio
import logging
from collections.abc import Mapping
from typing import Any, Callable

from langchain_core.messages import HumanMessage
from langfuse import propagate_attributes
from langfuse.langchain import CallbackHandler

from app.core.exceptions import AgentExecutionError
from app.core.config import settings


logger = logging.getLogger(__name__)


class AgentRequestService:
    def __init__(
        self,
        agent: Any,
        user_files: Mapping[str, str],
        callback_handler_factory: Callable[[], Any] = CallbackHandler,
    ):
        self._agent = agent
        self._user_files = user_files
        self._callback_handler_factory = callback_handler_factory

    async def run(self, user_input: str, user_id: str | None = None) -> str:
        has_document = bool(user_id and user_id in self._user_files)
        session_id = user_id or "anonymous"
        langfuse_handler = self._callback_handler_factory()

        try:
            with propagate_attributes(session_id=session_id):
                new_state = await asyncio.wait_for(
                    self._agent.ainvoke(
                        {
                            "messages": [HumanMessage(content=user_input)],
                            "user_id": user_id,
                            "has_document": has_document,
                        },
                        config={
                            "callbacks": [langfuse_handler],
                            "metadata": {
                                "user_id": user_id or "anonymous",
                                "has_document": str(has_document).lower(),
                            },
                            "tags": settings.tags,
                        },
                    ),
                    timeout=settings.task_timeout,
                )
                return str(new_state["messages"][-1].content)
        except Exception as exc:
            logger.exception("Agent error for user %s", user_id or "anonymous")
            raise AgentExecutionError("Error while using LLM-agent") from exc
