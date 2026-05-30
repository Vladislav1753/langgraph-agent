from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exceptions import (
    AgentExecutionError,
    DocumentNotFoundError,
    DocumentParseError,
    FileTooLargeError,
    UnsupportedFileFormatError,
)
from app.core.handlers import setup_exception_handlers


def create_test_client(exc: Exception) -> TestClient:
    app = FastAPI()
    setup_exception_handlers(app)

    @app.get("/raise")
    async def raise_error():
        raise exc

    return TestClient(app)


def test_object_not_found():
    client = create_test_client(DocumentNotFoundError("No document"))

    response = client.get("/raise")

    assert response.status_code == 404
    assert response.json() == {"detail": "No document"}


def test_file_too_large():
    client = create_test_client(FileTooLargeError("Too large"))

    response = client.get("/raise")

    assert response.status_code == 413
    assert response.json() == {"detail": "Too large"}


def test_unsupported_format():
    client = create_test_client(UnsupportedFileFormatError("Unsupported"))

    response = client.get("/raise")

    assert response.status_code == 415
    assert response.json() == {"detail": "Unsupported"}


def test_document_parse():
    client = create_test_client(DocumentParseError("Cannot parse"))

    response = client.get("/raise")

    assert response.status_code == 415
    assert response.json() == {"detail": "Cannot parse"}


def test_agent_execution():
    client = create_test_client(AgentExecutionError("Agent failed"))

    response = client.get("/raise")

    assert response.status_code == 503
    assert response.json() == {"detail": "Agent failed"}
