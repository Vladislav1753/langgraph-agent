import asyncio
from types import SimpleNamespace

import pytest

from app.services.agent import AgentRequestService
from app.core.exceptions import AgentExecutionError


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


def test_run_invokes_agent_without_document():
    agent = FakeAgent(response="general answer")
    callback_handler = object()
    service = AgentRequestService(
        agent, {}, callback_handler_factory=lambda: callback_handler
    )

    response = asyncio.run(service.run("hello"))

    assert response == "general answer"
    assert agent.last_state["user_id"] is None
    assert agent.last_state["has_document"] is False
    assert agent.last_config["callbacks"] == [callback_handler]
    assert agent.last_config["metadata"] == {
        "user_id": None,
        "has_document": False,
    }


def test_run_invokes_agent_with_cached_document():
    user_files = {"user-1": "Document text"}
    agent = FakeAgent(response="summary")
    callback_handler = object()
    service = AgentRequestService(
        agent, user_files, callback_handler_factory=lambda: callback_handler
    )

    response = asyncio.run(service.run("summarize", "user-1"))

    assert response == "summary"
    assert agent.last_state["user_id"] == "user-1"
    assert agent.last_state["has_document"] is True
    assert agent.last_state["messages"][0].content == "summarize"
    assert agent.last_config["callbacks"] == [callback_handler]
    assert agent.last_config["metadata"] == {
        "user_id": "user-1",
        "has_document": True,
    }
    assert agent.last_config["tags"] == ["development", "v1.0.0"]


def test_run_maps_agent_errors_to_service_error():
    service = AgentRequestService(
        FailingAgent(), {"user-1": "Document text"}, callback_handler_factory=object
    )

    with pytest.raises(AgentExecutionError):
        asyncio.run(service.run("summarize", "user-1"))
