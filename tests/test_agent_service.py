import asyncio
from types import SimpleNamespace

import pytest

from app.services.agent import AgentRequestService
from app.core.exceptions import AgentExecutionError, DocumentNotFoundError


class FakeAgent:
    def __init__(self, response: str = "ok"):
        self.response = response
        self.last_state = None
        self.last_config = None

    async def ainvoke(self, state, config=None):
        self.last_state = state
        self.last_config = config
        return {"messages": [SimpleNamespace(content=self.response)]}


class FailingAgent:
    async def ainvoke(self, state, config=None):
        raise RuntimeError("agent failed")


def test_run_raises_when_document_missing():
    service = AgentRequestService(FakeAgent(), {})

    with pytest.raises(DocumentNotFoundError):
        asyncio.run(service.run("summarize", "missing-user"))


def test_run_invokes_agent_with_cached_document():
    user_files = {"user-1": "Document text"}
    agent = FakeAgent(response="summary")
    callback_handler = object()
    service = AgentRequestService(
        agent, user_files, callback_handler_factory=lambda: callback_handler
    )

    response = asyncio.run(service.run("summarize", "user-1"))

    assert response == "summary"
    assert agent.last_state["text"] == "Document text"
    assert agent.last_state["user_id"] == "user-1"
    assert agent.last_state["messages"][0].content == "summarize"
    assert agent.last_config["callbacks"] == [callback_handler]
    assert agent.last_config["metadata"] == {"user_id": "user-1"}
    assert agent.last_config["tags"] == ["agent-request"]


def test_run_maps_agent_errors_to_service_error():
    service = AgentRequestService(
        FailingAgent(), {"user-1": "Document text"}, callback_handler_factory=object
    )

    with pytest.raises(AgentExecutionError):
        asyncio.run(service.run("summarize", "user-1"))
