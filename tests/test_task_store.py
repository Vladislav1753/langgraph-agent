from app.schemas.task import TaskStatus
from app.services.task_store import InMemoryTaskStore


def test_update_stores_result():
    store = InMemoryTaskStore()
    record = store.create("task-1")

    store.update("task-1", status=TaskStatus.ready, result="done")

    assert record.status == TaskStatus.ready
    assert record.result == "done"
    assert record.error is None


def test_update_stores_error():
    store = InMemoryTaskStore()
    record = store.create("task-1")

    store.update("task-1", status=TaskStatus.failed, error="boom")

    assert record.status == TaskStatus.failed
    assert record.result is None
    assert record.error == "boom"
