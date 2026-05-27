import pytest

from services.exceptions import (
    DocumentParseError,
    FileTooLargeError,
    UnsupportedFileFormatError,
)
from services.file_service import FileService


def test_upload_utf8_text_stores_document():
    user_files = {}
    service = FileService(user_files)

    result = service.upload("doc.txt", "text/plain", b"Hello document")

    assert result.status == "ok"
    assert result.length == len("Hello document")
    assert user_files[result.user_id] == "Hello document"


def test_upload_rejects_too_large_file():
    service = FileService({}, max_file_size=3)

    with pytest.raises(FileTooLargeError):
        service.upload("doc.txt", "text/plain", b"1234")


def test_upload_rejects_binary_text_file():
    service = FileService({})

    with pytest.raises(UnsupportedFileFormatError):
        service.upload("doc.bin", "application/octet-stream", b"\xff\xfe\x00")


def test_upload_rejects_invalid_pdf_without_caching():
    user_files = {}
    service = FileService(user_files)

    with pytest.raises(DocumentParseError):
        service.upload("bad.pdf", "application/pdf", b"not a pdf")

    assert user_files == {}


def test_upload_rejects_empty_text_without_caching():
    user_files = {}
    service = FileService(user_files)

    with pytest.raises(DocumentParseError):
        service.upload("empty.txt", "text/plain", b"   ")

    assert user_files == {}
