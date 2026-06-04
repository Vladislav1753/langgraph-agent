import time
from dataclasses import dataclass, field
from typing import MutableMapping

from cachetools import TTLCache

from app.schemas.task import TaskStatus


@dataclass
class TaskRecord:
    task_id: str
    status: TaskStatus = TaskStatus.pending
    result: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class InMemoryTaskStore:
    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        self._tasks: MutableMapping[str, TaskRecord] = TTLCache(
            maxsize=maxsize, ttl=ttl
        )

    def create(self, task_id: str) -> TaskRecord:
        record = TaskRecord(task_id=task_id)
        self._tasks[task_id] = record
        return record

    def get(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def update(
        self,
        task_id: str,
        *,
        status: TaskStatus,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        record = self._tasks.get(task_id)
        if not record:
            return
        record.status = status
        if result is not None:
            record.result = result
        if error is not None:
            record.error = error
        record.updated_at = time.time()
