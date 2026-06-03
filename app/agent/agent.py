import asyncio
import logging
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, TimeoutPolicy, default_retry_on
from langgraph.graph.message import add_messages
from app.agent.tools import browsing, help_tool, retrieving, text_agent
from app.agent.prompts import get_agent_chat_prompt
from app.core.config import settings

logger = logging.getLogger(__name__)

ALL_TOOLS = [browsing, retrieving, text_agent, help_tool]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str | None
    has_document: bool


def retry_on_transient_errors(exc: BaseException) -> bool:
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return False
    return default_retry_on(exc)


llm_retry = RetryPolicy(
    max_attempts=3,
    initial_interval=0.7,
    backoff_factor=2.0,
    max_interval=8.0,
    jitter=True,
    retry_on=retry_on_transient_errors,
)

llm_timeout = TimeoutPolicy(
    run_timeout=60,
    idle_timeout=25,
)


def create_llm_with_tools():
    primary = ChatDeepSeek(
        model="deepseek-v4-flash",
        temperature=0.2,
        max_tokens=None,
        timeout=30,
        max_retries=0,
        streaming=True,
    ).bind_tools(ALL_TOOLS)

    fallback = ChatOpenAI(
        model="gpt-5.4-mini-2026-03-17",
        temperature=0.2,
        max_tokens=None,
        timeout=30,
        max_retries=0,
        streaming=True,
    ).bind_tools(ALL_TOOLS)
    return primary.with_fallbacks([fallback])


def build_agent_graph(llm: Any | None = None) -> StateGraph:
    llm_with_tools = llm or create_llm_with_tools()
    tools_by_name = {tool.name: tool for tool in ALL_TOOLS}

    async def call_llm(state: AgentState) -> AgentState:
        chain = get_agent_chat_prompt() | llm_with_tools
        message = await chain.ainvoke(
            {
                "messages": list(state["messages"]),
                "user_id": state["user_id"],
                "has_document": state["has_document"],
            }
        )
        return {"messages": [message]}

    async def tool_node(state: AgentState) -> AgentState:
        tool_calls = getattr(state["messages"][-1], "tool_calls", []) or []
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            logger.info("Calling tool: %s", tool_name)
            try:
                if tool_name not in tools_by_name:
                    content = "Incorrect tool name. Please retry and select a tool from the available tools."
                else:
                    content = str(
                        await asyncio.wait_for(
                            tools_by_name[tool_name].ainvoke(tool_call["args"]),
                            timeout=settings.tool_timeout,
                        )
                    )

            except asyncio.TimeoutError:
                logger.warning("Tool %s timed out", tool_name)
                content = f"Tool '{tool_name}' timed out after {settings.tool_timeout}s. Try a different approach."
            except Exception as exc:
                logger.exception("Tool %s failed", tool_name)
                content = f"Tool '{tool_name}' failed: {exc}. Try a different tool or rephrase."

            results.append(
                ToolMessage(
                    tool_call_id=tool_call["id"], name=tool_name, content=content
                )
            )

        logger.info("Tool execution complete")
        return {"messages": results}

    def should_continue(state: AgentState) -> bool:
        result = state["messages"][-1]
        tool_calls = getattr(result, "tool_calls", []) or []
        return len(tool_calls) > 0

    graph = StateGraph(AgentState)
    graph.add_node(
        "agent",
        call_llm,
        retry_policy=llm_retry,
        timeout=llm_timeout,
    )
    graph.add_edge(START, "agent")
    graph.add_node("tool_node", tool_node)
    graph.add_conditional_edges(
        "agent", should_continue, {True: "tool_node", False: END}
    )
    graph.add_edge("tool_node", "agent")

    return graph
