import argparse
import json
import logging
import re
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langfuse import Evaluation, get_client

from app.agent.agent import build_agent_graph
from app.agent.tools import ingest_document, set_document_store
from app.services.agent import AgentRequestService


logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """
You are an impartial LLM-as-a-judge evaluator.
Compare the expected answer and the actual answer for the same user input.

Return only valid JSON with this schema:
{
  "score": 0.0,
  "comment": "short explanation"
}

Scoring rubric:
- 1.0: The actual answer is semantically equivalent to the expected answer.
- 0.8: Mostly correct, with only minor omissions or wording differences.
- 0.5: Partially correct, but misses important requirements or details.
- 0.2: Mostly incorrect, barely related, or unsupported by the expected answer.
- 0.0: Completely incorrect, empty, or contradicts the expected answer.

Do not require exact wording. Judge semantic correctness, grounding, and whether
the actual answer satisfies the user request. Penalize hallucinated document
claims when the expected answer says no document is available.
""".strip()

user_files: dict[str, str] = {}
ingested_user_ids: set[str] = set()
service: AgentRequestService | None = None
judge_llm: ChatDeepSeek | None = None


def _expected_answer(expected_output: Any) -> str:
    if isinstance(expected_output, dict):
        answer = expected_output.get("answer")
        if isinstance(answer, str):
            return answer

    if isinstance(expected_output, str):
        return expected_output

    return json.dumps(expected_output, ensure_ascii=False, indent=2)


def _parse_judge_response(content: str) -> tuple[float, str]:
    text = content.strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return 0.0, f"Judge returned non-JSON output: {text[:300]}"
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return 0.0, f"Judge returned malformed JSON: {text[:300]}"

    raw_score = payload.get("score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0

    score = max(0.0, min(1.0, score))
    comment = str(payload.get("comment", "")).strip() or "No judge comment."
    return score, comment


async def llm_judge_semantic_match(
    *,
    input: Any,
    output: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Evaluation:
    if judge_llm is None:
        raise RuntimeError("Judge LLM is not initialized.")

    expected = _expected_answer(expected_output)
    actual = str(output)

    if not expected.strip():
        return Evaluation(
            name="llm_judge_semantic_match",
            value=0.0,
            comment="Expected output is empty, cannot judge semantic match.",
            data_type="NUMERIC",
        )

    judge_prompt = f"""
User input:
{json.dumps(input, ensure_ascii=False, indent=2)}

Dataset metadata:
{json.dumps(metadata or {}, ensure_ascii=False, indent=2)}

Expected answer:
{expected}

Actual answer:
{actual}
""".strip()

    message = await judge_llm.ainvoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=judge_prompt),
        ]
    )
    score, comment = _parse_judge_response(str(message.content))

    return Evaluation(
        name="llm_judge_semantic_match",
        value=score,
        comment=comment,
        data_type="NUMERIC",
        metadata={
            "judge_model": "deepseek-v4-flash",
            "expected_answer": expected,
        },
    )


async def task(*, item: Any, **kwargs: Any) -> str:
    if service is None:
        raise RuntimeError("Agent service is not initialized.")

    item_input = item.input
    user_input = item_input["user_input"]
    document_text = item_input.get("document_text")
    user_id = item_input.get("user_id")

    if document_text:
        user_id = user_id or f"eval-{item.id}"
        user_files[user_id] = document_text

        if user_id not in ingested_user_ids:
            logger.info("Indexing eval document for user_id=%s", user_id)
            await ingest_document(user_id, document_text)
            ingested_user_ids.add(user_id)

    return await service.run(user_input=user_input, user_id=user_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Langfuse offline evaluation with an LLM-as-a-judge evaluator."
    )
    parser.add_argument(
        "--dataset",
        default="langgraph-agent-eval-v3",
        help="Langfuse dataset name.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max concurrent eval items. Keep low to avoid Pinecone and LLM rate limits.",
    )
    parser.add_argument(
        "--show-items",
        action="store_true",
        help="Print individual item results.",
    )
    return parser.parse_args()


def main() -> None:
    global judge_llm, service

    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    load_dotenv(".env")

    set_document_store(user_files)
    agent = build_agent_graph().compile()
    service = AgentRequestService(agent=agent, user_files=user_files)
    judge_llm = ChatDeepSeek(
        model="deepseek-v4-flash",
        temperature=0.0,
        max_tokens=500,
        timeout=None,
        max_retries=2,
        streaming=False,
    )

    langfuse = get_client()
    dataset = langfuse.get_dataset(args.dataset)

    experiment_name = f"agent-llm-judge-eval-{datetime.now().isoformat()}"
    result = dataset.run_experiment(
        name=experiment_name,
        description="Offline evaluation for current agent version using LLM-as-a-judge.",
        task=task,
        evaluators=[llm_judge_semantic_match],
        max_concurrency=args.concurrency,
        metadata={
            "agent_version": "local",
            "model": "deepseek-v4-flash",
            "judge_model": "deepseek-v4-flash",
        },
    )

    print(result.format(include_item_results=args.show_items))
    langfuse.flush()


if __name__ == "__main__":
    main()
