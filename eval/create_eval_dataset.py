import argparse
import os
from datetime import date

from dotenv import load_dotenv
from langfuse import get_client


EVAL_DOCUMENT_TEXT = """
UV Project Cheatsheet

Purpose:
uv is a fast Python package and project manager. It can create projects, manage
dependencies, keep an environment synchronized with a lockfile, and run commands
inside the project environment.

Setup commands:
- uv init creates a new Python project.
- uv add <package> adds a runtime dependency.
- uv add --dev <package> adds a development dependency.
- uv sync installs dependencies from pyproject.toml and uv.lock.
- uv lock updates the lockfile.
- uv run <command> runs a command inside the managed environment.

Development workflow:
Use uv sync after cloning a repository so the local environment matches uv.lock.
Use uv run pytest to run tests. Keep uv.lock committed so team members and CI use
the same dependency versions.

Docker deployment:
For containers, copy pyproject.toml and uv.lock first, run uv sync --frozen, then
copy application code. Use uvicorn app.main:app --host 0.0.0.0 --port 8000 for a
FastAPI application. Do not use unpinned latest images for infrastructure services
such as databases.

Best practices:
Separate runtime and development dependencies, commit lockfiles, use frozen
installs in CI and Docker, pin infrastructure image versions, and keep secrets in
environment variables rather than source code.
""".strip()


DATASET_ITEMS = [
    {
        "input": {"user_input": "What can you do?"},
        "expected_output": {
            "answer": "I can answer general questions, search the web for current external information, and work with uploaded documents by retrieving relevant chunks, summarizing them, answering document-specific questions, and generating follow-up questions."
        },
        "metadata": {
            "category": "help",
            "requires_document": False,
            "criteria": "Explains general chat, web search, document Q&A, summarization, and question generation capabilities.",
        },
    },
    {
        "input": {
            "user_input": "Summarize the uploaded document.",
            "has_document": True,
            "document_text": EVAL_DOCUMENT_TEXT,
        },
        "expected_output": {
            "answer": "The document is a uv project cheatsheet. It explains that uv is a fast Python package and project manager for creating projects, managing dependencies, syncing environments from pyproject.toml and uv.lock, and running commands. It lists setup commands such as uv init, uv add, uv add --dev, uv sync, uv lock, and uv run. It recommends committing uv.lock, using uv run pytest for tests, using uv sync --frozen in Docker or CI, starting FastAPI with uvicorn app.main:app, pinning infrastructure image versions, and keeping secrets in environment variables."
        },
        "metadata": {
            "category": "summary",
            "requires_document": True,
            "criteria": "The answer should be grounded only in the uploaded document and avoid unsupported details.",
        },
    },
    {
        "input": {
            "user_input": "Generate five questions from the uploaded document.",
            "has_document": True,
            "document_text": EVAL_DOCUMENT_TEXT,
        },
        "expected_output": {
            "answer": "1. What is uv used for in a Python project?\n2. Which command installs dependencies from pyproject.toml and uv.lock?\n3. How do you add a runtime dependency and a development dependency with uv?\n4. Why should uv.lock be committed to the repository?\n5. What Docker practices does the document recommend for a FastAPI application?"
        },
        "metadata": {
            "category": "questions",
            "requires_document": True,
            "criteria": "The answer should contain exactly five relevant questions based on the uploaded document.",
        },
    },
    {
        "input": {
            "user_input": "What are the main takeaways from the document?",
            "has_document": True,
            "document_text": EVAL_DOCUMENT_TEXT,
        },
        "expected_output": {
            "answer": "The main takeaways are: uv manages Python projects and dependencies; uv sync keeps the environment aligned with pyproject.toml and uv.lock; uv run executes commands such as pytest inside the managed environment; uv.lock should be committed for reproducibility; Docker and CI should use uv sync --frozen; FastAPI can be started with uvicorn app.main:app; infrastructure images should be pinned instead of using latest; and secrets should be stored in environment variables."
        },
        "metadata": {
            "category": "retrieval",
            "requires_document": True,
            "criteria": "The answer should use retrieval and summarize the most relevant document points.",
        },
    },
    {
        "input": {
            "user_input": "Find the section that explains setup commands.",
            "has_document": True,
            "document_text": EVAL_DOCUMENT_TEXT,
        },
        "expected_output": {
            "answer": "The setup commands section lists: uv init to create a new Python project; uv add <package> to add a runtime dependency; uv add --dev <package> to add a development dependency; uv sync to install dependencies from pyproject.toml and uv.lock; uv lock to update the lockfile; and uv run <command> to run a command inside the managed environment."
        },
        "metadata": {
            "category": "retrieval",
            "requires_document": True,
            "criteria": "The answer should retrieve setup-related content and avoid unrelated sections.",
        },
    },
    {
        "input": {
            "user_input": "Compare this document with current best practices online.",
            "has_document": True,
            "document_text": EVAL_DOCUMENT_TEXT,
        },
        "expected_output": {
            "answer": "Document-based points: the document recommends committing uv.lock, using uv sync after cloning, running tests with uv run pytest, using uv sync --frozen in Docker or CI, starting FastAPI with uvicorn app.main:app, pinning infrastructure image versions, and storing secrets in environment variables. Current best practices are consistent with those points: reproducible lockfile installs, frozen CI/container builds, separated dev/runtime dependencies, pinned infrastructure images, and environment-based secrets are all standard recommendations."
        },
        "metadata": {
            "category": "hybrid",
            "requires_document": True,
            "criteria": "The answer should use both document retrieval and web search while clearly separating sources.",
        },
    },
    {
        "input": {
            "user_input": "Search the web for similar articles about RAG evaluation."
        },
        "expected_output": {
            "answer": "Relevant resources for RAG evaluation include the Langfuse evaluation and datasets documentation, RAGAS documentation for faithfulness and context relevance metrics, DeepEval documentation for RAG and hallucination metrics, and guides about measuring retrieval quality, answer faithfulness, context precision, context recall, and hallucination risk."
        },
        "metadata": {
            "category": "web_search",
            "requires_document": False,
            "criteria": "The answer should use web search and return relevant external resources.",
        },
    },
    {
        "input": {
            "user_input": "Answer this from my document: what commands are listed?",
            "has_document": True,
            "document_text": EVAL_DOCUMENT_TEXT,
        },
        "expected_output": {
            "answer": "The document lists these commands: uv init, uv add <package>, uv add --dev <package>, uv sync, uv lock, uv run <command>, uv run pytest, and uvicorn app.main:app --host 0.0.0.0 --port 8000."
        },
        "metadata": {
            "category": "retrieval",
            "requires_document": True,
            "criteria": "The answer should list only commands present in the uploaded document.",
        },
    },
    {
        "input": {
            "user_input": "Summarize my document and then create three follow-up questions.",
            "has_document": True,
            "document_text": EVAL_DOCUMENT_TEXT,
        },
        "expected_output": {
            "answer": "Summary: The document is a uv project cheatsheet that explains uv's role in Python project management, dependency installation, lockfile-based synchronization, testing, and Docker deployment. It recommends committing uv.lock, using uv sync --frozen in CI and Docker, pinning infrastructure image versions, and storing secrets in environment variables.\n\nFollow-up questions:\n1. When should you use uv sync instead of uv lock?\n2. Why is uv sync --frozen useful in Docker or CI?\n3. What risks come from using latest tags for infrastructure images?"
        },
        "metadata": {
            "category": "summary_questions",
            "requires_document": True,
            "criteria": "The answer should include a summary and exactly three follow-up questions.",
        },
    },
    {
        "input": {
            "user_input": "What does my uploaded document say about deployment?",
            "has_document": False,
        },
        "expected_output": {
            "answer": "I do not have an uploaded document available for this request. Please upload a document first, or ask a question that does not require document context."
        },
        "metadata": {
            "category": "missing_document",
            "requires_document": False,
            "criteria": "The answer should not pretend a document is available and should explain the limitation.",
        },
    },
]


def create_dataset(dataset_name: str, description: str) -> None:
    load_dotenv()
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env or environment."
        )

    langfuse = get_client()
    langfuse.create_dataset(
        name=dataset_name,
        description=description,
        metadata={
            "author": "langgraph-agent",
            "date": date.today().isoformat(),
            "type": "benchmark",
        },
    )

    for item in DATASET_ITEMS:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=item["input"],
            expected_output=item["expected_output"],
            metadata=item["metadata"],
        )

    langfuse.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Langfuse evaluation dataset."
    )
    parser.add_argument(
        "--name",
        default="langgraph-agent-eval-v3",
        help="Langfuse dataset name.",
    )
    parser.add_argument(
        "--description",
        default="Evaluation dataset for the LangGraph document agent.",
        help="Langfuse dataset description.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_dataset(args.name, args.description)
    print(f"Created Langfuse dataset '{args.name}' with {len(DATASET_ITEMS)} items.")
