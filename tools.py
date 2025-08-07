from langchain_core.tools import tool

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from dotenv import load_dotenv
from pinecone import Pinecone
from doc_loader import extract_text_pdf
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List


@tool
async def browsing(query: str) -> str:
    """Browse a 'query' in  DuckDuckGo browser to find similar documents."""
    wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", max_results=5)
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