import logging
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from tools import browsing, help_tool, ingesting, retrieving, text_agent


logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """
You are an intelligent AI agent that works with articles and documents.
Your task is to choose a correct tool provided and make a final answer after completing all tasks.
You can make multiple calls if needed. Do not add any greetings or extra comments.
Always cite the specific parts of the documents you use in your answers.

Available tools:
- 'browsing': Search DuckDuckGo for up-to-date information or documents.
- 'ingesting': Split and store the user's document in a vector database. Call this first before 'retrieving'.
- 'retrieving': Search stored documents semantically. Requires user_id and a search query. Call after 'ingesting'.
- 'text_agent': Generates a summary and/or questions about the provided document.
- 'help_tool': Describes what the agent can currently do.

Use 'ingesting' first if the user wants to search within their document.
Use 'retrieving' after ingestion to answer specific questions about the document.
Use 'browsing' only when the user asks for similar articles or external information.
Use 'text_agent' when the user asks for a summary or generated questions.
Use 'help_tool' when the user asks about your functionality.
"""

ALL_TOOLS = [browsing, ingesting, retrieving, text_agent, help_tool]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    text: str
    user_id: str


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
        system_prompt = SystemMessage(content=AGENT_SYSTEM_PROMPT)
        doc_message = SystemMessage(
            content=f"Document provided by user:\n\n{state['text']}"
        )

        messages = [system_prompt, doc_message] + list(state["messages"])
        message = await llm_with_tools.ainvoke(messages)
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
