import logging
import uuid

from app.schemas.task import TaskStatus
from app.services.agent import AgentRequestService
from app.services.task_store import InMemoryTaskStore, TaskRecord

logger = logging.getLogger(__name__)


class TaskService:
    def __init__(self, agent_service: AgentRequestService, store: InMemoryTaskStore):
        self._agent_service = agent_service
        self._store = store

    def create_task(self) -> str:
        task_id = uuid.uuid4().hex
        self._store.create(task_id)
        return task_id

    def get_task(self, task_id: str) -> TaskRecord | None:
        return self._store.get(task_id)

    async def run_task(
        self, task_id: str, user_input: str, user_id: str | None
    ) -> None:
        self._store.update(task_id=task_id, status=TaskStatus.processing)
        try:
            result = await self._agent_service.run(
                user_input=user_input, user_id=user_id
            )
            self._store.update(task_id=task_id, status=TaskStatus.ready, result=result)
        except Exception as exc:
            logger.exception("Task %s failed", task_id)
            self._store.update(task_id, status=TaskStatus.failed, error=str(exc))
