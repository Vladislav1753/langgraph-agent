from http import HTTPStatus

from fastapi import APIRouter, BackgroundTasks, Request

from app.core.exceptions import TaskNotFoundError
from app.schemas.task import (
    TaskCreateRequest,
    TaskCreatedResponse,
    TaskStatus,
    TaskStatusResponse,
)

router = APIRouter()


@router.post("/", response_model=TaskCreatedResponse, status_code=HTTPStatus.ACCEPTED)
async def create_task(
    request: Request,
    payload: TaskCreateRequest,
    background_tasks: BackgroundTasks,
):
    task_service = request.app.state.task_service.run_task
    task_id = request.app.state.task_service.create_task()
    background_tasks.add_task(
        task_service, task_id, payload.user_input, payload.user_id
    )

    return TaskCreatedResponse(task_id=task_id, status=TaskStatus.pending)


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task(request: Request, task_id: str):
    record = request.app.state.task_service.get_task(task_id)
    if record is None:
        raise TaskNotFoundError(f"Task {task_id} not found")

    return TaskStatusResponse(
        task_id=task_id, status=record.status, result=record.result, error=record.error
    )
