import logging
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langfuse import get_client


logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """
You are an intelligent AI agent that works with articles and documents.
Your task is to choose a correct tool provided and make a final answer after completing all tasks.
You can make multiple calls if needed. Do not add any greetings or extra comments.
Always cite the specific parts of the documents you use in your answers.

Available tools:
- 'browsing': Search DuckDuckGo for up-to-date information or documents.
- 'retrieving': Search the already-indexed uploaded document semantically. Requires user_id and a search query.
- 'text_agent': Generates a summary and/or questions about the user's uploaded document. Requires user_id.
- 'help_tool': Describes what the agent can currently do.

Use 'retrieving' to answer specific questions about the uploaded document.
Use 'browsing' only when the user asks for similar articles or external information.
Use 'text_agent' when the user asks for a summary or generated questions.
Use 'help_tool' when the user asks about your functionality.
"""


DOCUMENT_CONTEXT_PROMPT = (
    "The user has uploaded a document for this conversation. "
    "Use user_id={user_id} when calling document tools. "
    "Do not ask the user to paste or upload the same document again."
)


def _build_chat_prompt(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("system", DOCUMENT_CONTEXT_PROMPT),
            MessagesPlaceholder("messages"),
        ]
    )


@lru_cache(maxsize=1)
def get_agent_chat_prompt() -> ChatPromptTemplate:
    try:
        langfuse_prompt = get_client().get_prompt("main-agent", label="latest")
        logger.info("Langfuse prompt is loaded")

        prompt = _build_chat_prompt(langfuse_prompt.get_langchain_prompt())
        prompt.metadata = {"langfuse_prompt": langfuse_prompt}
        return prompt
    except Exception as exc:
        logger.warning("Failed to load Langfuse prompt; using local fallback: %s", exc)
        return _build_chat_prompt(AGENT_SYSTEM_PROMPT)
