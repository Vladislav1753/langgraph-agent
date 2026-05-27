import asyncio
import logging
from typing import Any

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone


logger = logging.getLogger(__name__)

INDEX_NAME = "doc-index"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_EMBED_MODEL = "llama-text-embed-v2"
PINECONE_RERANK_MODEL = "bge-reranker-v2-m3"
PINECONE_TEXT_FIELD = "chunk_text"


@tool
async def browsing(query: str, max_results: int = 5) -> str:
    """Browse a query in DuckDuckGo to find similar documents."""
    return await asyncio.to_thread(_browse_sync, query, max_results)


def _browse_sync(query: str, max_results: int) -> str:
    wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", max_results=max_results)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    return search.invoke(query)


def _get_or_create_index():
    pc = Pinecone()
    if not _has_index(pc, INDEX_NAME):
        _create_integrated_index(pc, INDEX_NAME)
    return _index(pc, INDEX_NAME)


def _has_index(pc: Pinecone, name: str) -> bool:
    indexes_api = getattr(pc, "indexes", None)
    if indexes_api and hasattr(indexes_api, "exists"):
        return indexes_api.exists(name)
    return pc.has_index(name)


def _create_integrated_index(pc: Pinecone, name: str) -> None:
    indexes_api = getattr(pc, "indexes", None)
    if indexes_api and hasattr(indexes_api, "create"):
        try:
            from pinecone.models.indexes.specs import EmbedConfig, IntegratedSpec

            indexes_api.create(
                name=name,
                spec=IntegratedSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION,
                    embed=EmbedConfig(
                        model=PINECONE_EMBED_MODEL,
                        field_map={"text": PINECONE_TEXT_FIELD},
                    ),
                ),
            )
            return
        except ImportError:
            logger.info(
                "Falling back to create_index_for_model for Pinecone SDK compatibility"
            )

    pc.create_index_for_model(
        name=name,
        cloud=PINECONE_CLOUD,
        region=PINECONE_REGION,
        embed={
            "model": PINECONE_EMBED_MODEL,
            "field_map": {"text": PINECONE_TEXT_FIELD},
        },
    )


def _index(pc: Pinecone, name: str):
    index_factory = getattr(pc, "index", None)
    if index_factory:
        return index_factory(name)
    return pc.Index(name)


def _upsert_records(namespace: str, records: list[dict[str, str]]) -> None:
    index = _get_or_create_index()
    index.upsert_records(namespace=namespace, records=records)


def _search_records(namespace: str, query: str) -> Any:
    index = _get_or_create_index()
    rerank = {
        "model": PINECONE_RERANK_MODEL,
        "top_n": 5,
        "rank_fields": [PINECONE_TEXT_FIELD],
    }

    try:
        return index.search(
            namespace=namespace,
            top_k=5,
            inputs={"text": query},
            rerank=rerank,
        )
    except TypeError:
        return index.search(
            namespace=namespace,
            query={
                "top_k": 5,
                "inputs": {"text": query},
            },
            rerank=rerank,
        )


def _get_hits(results: Any) -> list[Any]:
    if not results:
        return []

    if isinstance(results, dict):
        return results.get("result", {}).get("hits", [])

    result = getattr(results, "result", None)
    return list(getattr(result, "hits", []) or [])


def _get_hit_text(hit: Any) -> str:
    if isinstance(hit, dict):
        fields = hit.get("fields", {})
    else:
        fields = getattr(hit, "fields", {}) or {}

    return fields.get(PINECONE_TEXT_FIELD, "")


@tool
async def ingesting(text: str, user_id: str) -> str:
    """Ingest document chunks into Pinecone for semantic search."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    docs_to_upsert = [
        {"_id": f"chunk-{i}", PINECONE_TEXT_FIELD: chunk}
        for i, chunk in enumerate(chunks)
    ]

    await asyncio.to_thread(_upsert_records, user_id, docs_to_upsert)
    await asyncio.sleep(2)
    return (
        f"Successfully ingested {len(chunks)} chunks for user {user_id}. "
        "You can now use 'retrieving' to search this document."
    )


@tool
async def retrieving(user_id: str, query: str) -> str:
    """Perform semantic search over stored documents."""
    results = await asyncio.to_thread(_search_records, user_id, query)
    hits = _get_hits(results)

    if not hits:
        return "No relevant information found in the document for this query."

    return "\n\n".join(
        f"[Chunk {i + 1}] {_get_hit_text(hit)}" for i, hit in enumerate(hits)
    )


@tool
async def text_agent(text: str, task: str, user_id: str, n_questions: int = 5) -> str:
    """Generate summaries and/or questions for a document."""
    if task not in ["summary", "questions", "both"]:
        return "Wrong task choice, provide one of the available tasks: 'summary', 'questions', 'both'."

    logger.info("User %s called text_agent task %s", user_id, task)

    text_llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        streaming=True,
    )

    system_prompt = SystemMessage(
        content="""
You are a specialized agent for document processing.
Your role is limited to two tasks:
1. Generate a concise and accurate summary of the given text.
2. Generate the requested number of questions based on the content of the text.
Rules:
- Only perform the task(s) requested (summary, questions).
- Do not add any greetings, explanations, or extra comments.
- Use only the input text; do not invent or hallucinate.
- The output must be concise, structured, and strictly limited to the task.
"""
    )

    doc_message = SystemMessage(content=f"Document provided by user:\n\n{text}.")
    human_message = HumanMessage(
        content=f"Your task: {task}. Number of questions: {n_questions}"
    )

    try:
        message = await text_llm.ainvoke([system_prompt, doc_message, human_message])
    except Exception as exc:
        logger.exception("Text agent error for user %s", user_id)
        return f"Error {exc} while invoking the llm"

    return message.content


@tool
def help_tool(user_id: str) -> str:
    """Return information about the agent's current functionality."""
    logger.info("User %s asks about functionality", user_id)
    return """
Here is what I can currently do:
- Answer your questions about documents you've uploaded using semantic search.
- Search the web for similar documents or updated information.
- Summarize your documents and generate follow-up questions.
"""
