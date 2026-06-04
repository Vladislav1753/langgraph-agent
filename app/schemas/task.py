from enum import Enum
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class TaskCreateRequest(BaseModel):
    user_input: str = Field(..., min_length=1)
    user_id: str | None = None


class TaskCreatedResponse(BaseModel):
    task_id: str
    status: TaskStatus


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: str | None = None
    error: str | None = None
