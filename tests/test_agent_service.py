import asyncio
from types import SimpleNamespace

import pytest

from services.agent_service import AgentRequestService
from services.exceptions import AgentExecutionError, DocumentNotFoundError


class FakeAgent:
    def __init__(self, response: str = "ok"):
        self.response = response
        self.last_state = None

    async def ainvoke(self, state):
        self.last_state = state
        return {"messages": [SimpleNamespace(content=self.response)]}


class FailingAgent:
    async def ainvoke(self, state):
        raise RuntimeError("agent failed")


def test_run_raises_when_document_missing():
    service = AgentRequestService(FakeAgent(), {})

    with pytest.raises(DocumentNotFoundError):
        asyncio.run(service.run("summarize", "missing-user"))


def test_run_invokes_agent_with_cached_document():
    user_files = {"user-1": "Document text"}
    agent = FakeAgent(response="summary")
    service = AgentRequestService(agent, user_files)

    response = asyncio.run(service.run("summarize", "user-1"))

    assert response == "summary"
    assert agent.last_state["text"] == "Document text"
    assert agent.last_state["user_id"] == "user-1"
    assert agent.last_state["messages"][0].content == "summarize"


def test_run_maps_agent_errors_to_service_error():
    service = AgentRequestService(FailingAgent(), {"user-1": "Document text"})

    with pytest.raises(AgentExecutionError):
        asyncio.run(service.run("summarize", "user-1"))
