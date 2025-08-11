from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from tools import browsing, retrieving, ingesting, text_agent, help_tool
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from doc_loader import extract_text_pdf

import asyncio


load_dotenv()

all_tools = [browsing, ingesting, retrieving, text_agent, help_tool]
all_tools_d = {tool.name: tool for tool in all_tools}

# probably 'deepseek-reasoner' is a better choice for an agent, but let's stick with 'chat' for now
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    streaming=True).bind_tools(all_tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    text: str
    user_id: int

async def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM-agent with the current state"""

    system_prompt = SystemMessage(content="""
    You are an intelligent AI agent that works with articles and documents. 
    Your task is to choose a correct tool provided and make a final answer after completing all tasks.
    You can make multiple calls if needed. Do not add any greetings or extra comments.
    Always cite the specific parts of the documents you use in your answers.
    
    Available tools:
    - 'browsing': Search DuckDuckGo for up-to-date information or documents.
    - 'ingesting': Split and store documents in a vector database.
    - 'retrieving': Search stored documents semantically when the user asks questions about ingested documents.
    - 'text_agent': Generates a summary and/or questions about the provided document.
    - 'help_tool': Describes what the agent can currently do.
    
    Use 'ingesting' to add documents if the user provides a document **and** has questions or wants to search within it.  
    Use 'retrieving' to answer questions about documents already stored.  
    Use 'browsing' only when user asks for similar articles or documents.
    Use 'text_agent' when user asks for summarization or questions based on the document.
    Use 'help_tool' when user has any questions about your functionality. 
    """)

    doc_message = SystemMessage(content=f"Document provided by user: \n\n{state['text']}")

    messages = [system_prompt, doc_message] + list(state['messages'])
    message = await llm.ainvoke(messages)
    return {'messages': message}

async def tool_node(state: AgentState) -> AgentState:
    """Execute tool calls from LLM's response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']}")

        if not t['name'] in all_tools_d:
            print(F"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = await all_tools_d[t['name']].ainvoke(t['args'])

        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

def should_continue(state: AgentState):
    """Check if the last message contain tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


graph = StateGraph(AgentState)
graph.add_node("agent", call_llm)
graph.add_edge(START, "agent")
graph.add_node("tool_node", tool_node)

graph.add_conditional_edges(
    "agent",
    should_continue,
    {True: "tool_node", False: END}
)

graph.add_edge("tool_node", "agent")

# Uncomment to be able to run it in CLI
# agent = graph.compile()
#
# # document I used
# doc_path = "data/Understanding LSTM Networks -- colah's blog.pdf"
# text = extract_text_pdf(doc_path)
# # using only first 3000 symbols for resource economy purposes
# text = text[:3000]
#
# async def running_agent():
#     print("\n=== ARTICLE AGENT ===")
#
#     while True:
#         user_input = input("\nUser's input: ")
#         if user_input.lower() in ['exit', 'quit']:
#             break
#         messages = [HumanMessage(content=user_input)]
#
#         result = await agent.ainvoke({"messages": messages, "text": text, "user_id": 1})
#
#         print("\n=== ANSWER ===")
#         print(result['messages'][-1].content)
#
# if __name__ == "__main__":
#     asyncio.run(running_agent())
