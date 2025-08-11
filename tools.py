from langchain_core.tools import tool

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from pinecone import Pinecone
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage, SystemMessage


@tool
async def browsing(query: str, max_results: int = 5) -> str:
    """Browse a 'query' in  DuckDuckGo browser to find similar documents."""
    wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", max_results=max_results)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)

    return search.invoke(query)


@tool
async def ingesting(text: str, user_id: int) -> str:
    """Ingests document chunks into a vector database for semantic search."""
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
    chunks = splitter.split_text(text)

    docs_to_upsert = [
            {"id": f"chunk-{i}", "chunk_text": chunk}
            for i, chunk in enumerate(chunks)
        ]

    pc = Pinecone()

    index_name = "doc-index"
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud='aws',
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
            }
        )

    dense_index = pc.Index(index_name)

    dense_index.upsert_records(namespace=user_id, records=docs_to_upsert)
    time.sleep(2)
    return dense_index


@tool
async def retrieving(user_id: int, query: str, dense_index):
    """Performs semantic search over stored documents."""
    results = dense_index.search(
        namespace=user_id,
        query={
            "top_k": 5,
            "inputs": {
                'text': query
            }
        },
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": 5,
            "rank_fields": ["chunk_text"]
        }
    )

    if not results:
        return "I found no relevant information in this text."

    return results


@tool
async def text_agent(text: str, task: str, user_id: int, n_questions: int = 5) -> str:
    """Text agent tool that generates questions and summarizes documents

    Args:
    'text' - text that AI will work with
    'task' - can be "summary", "questions", "both"
    'n_questions' - number of questions asked by the user, defaults to 5
    'user_id' - user id
    """

    if task not in ["summary", "questions", "both"]:
        return "Wrong task choice, provide one of the available tasks: 'summary', 'questions', 'both'."

    print(f"User {user_id} called task {task}")

    text_llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        streaming=True)

    system_prompt = SystemMessage(content="""
    You are a specialized agent for document processing.
    Your role is limited to two tasks:
    1. Generate a concise and accurate summary of the given text.
    2. Generate 5–7 questions based on the content of the text.
    Rules:
    - Only perform the task(s) requested (summary, questions).
    - Do not add any greetings, explanations, or extra comments.
    - Use only the input text; do not invent or hallucinate.
    - The output must be concise, structured, and strictly limited to the task.
    """)

    doc_message = SystemMessage(content=f"Document provided by user: \n\n{text}.")
    human_message = HumanMessage(content=f"Your task: {task}")

    messages = [system_prompt, doc_message, human_message]
    try:
        message = await text_llm.ainvoke(messages)
    except Exception as e:
        return f"Error {e} while invoking the llm"
    return message.content


@tool
def help_tool(user_id: int) -> str:
    """Tool that gives user information about agent's current functionality"""

    print(f"User {user_id} asks about functionality")
    functionality = """
    Here’s what I can currently do:
    - Answer your questions about documents you've uploaded using semantic search.
    - Search the web for similar documents or updated information.
    - Summarize your documents and generate follow-up questions.
    """
    return functionality
