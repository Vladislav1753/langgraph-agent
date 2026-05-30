import asyncio

import pytest

from app.core.exceptions import (
    DocumentParseError,
    FileTooLargeError,
    UnsupportedFileFormatError,
)
from app.services.file import FileService


def test_upload_utf8_text_stores_document():
    user_files = {}

    async def fake_ingest(user_id: str, text: str) -> int:
        assert user_id
        assert text == "Hello document"
        return 1

    service = FileService(user_files, ingest_document_func=fake_ingest)
    result = asyncio.run(service.upload("doc.txt", "text/plain", b"Hello document"))

    assert result.status == "ok"
    assert result.length == len("Hello document")
    assert user_files[result.user_id] == "Hello document"


def test_upload_rejects_too_large_file():
    service = FileService({}, max_file_size=3)

    with pytest.raises(FileTooLargeError):
        asyncio.run(service.upload("doc.txt", "text/plain", b"1234"))


def test_upload_rejects_binary_text_file():
    service = FileService({})

    with pytest.raises(UnsupportedFileFormatError):
        asyncio.run(
            service.upload("doc.bin", "application/octet-stream", b"\xff\xfe\x00")
        )


def test_upload_rejects_invalid_pdf_without_caching():
    user_files = {}
    service = FileService(user_files)

    with pytest.raises(DocumentParseError):
        asyncio.run(service.upload("bad.pdf", "application/pdf", b"not a pdf"))

    assert user_files == {}


def test_upload_rejects_empty_text_without_caching():
    user_files = {}
    service = FileService(user_files)

    with pytest.raises(DocumentParseError):
        asyncio.run(service.upload("empty.txt", "text/plain", b"   "))

    assert user_files == {}
