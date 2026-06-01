import logging
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from app.agent.tools import browsing, help_tool, retrieving, text_agent
from app.agent.prompts import get_agent_chat_prompt

logger = logging.getLogger(__name__)

ALL_TOOLS = [browsing, retrieving, text_agent, help_tool]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str | None
    has_document: bool


def create_llm_with_tools():
    return ChatDeepSeek(
        model="deepseek-v4-flash",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        streaming=True,
    ).bind_tools(ALL_TOOLS)


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

            if tool_name not in tools_by_name:
                result = "Incorrect tool name. Please retry and select a tool from the available tools."
            else:
                result = await tools_by_name[tool_name].ainvoke(tool_call["args"])

            results.append(
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                    content=str(result),
                )
            )

        logger.info("Tool execution complete")
        return {"messages": results}

    def should_continue(state: AgentState) -> bool:
        result = state["messages"][-1]
        tool_calls = getattr(result, "tool_calls", []) or []
        return len(tool_calls) > 0

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_llm)
    graph.add_edge(START, "agent")
    graph.add_node("tool_node", tool_node)
    graph.add_conditional_edges(
        "agent", should_continue, {True: "tool_node", False: END}
    )
    graph.add_edge("tool_node", "agent")

    return graph
