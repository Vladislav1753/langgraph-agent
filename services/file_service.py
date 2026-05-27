import logging
import uuid
from collections.abc import MutableMapping
from dataclasses import dataclass

from config import MAX_FILE_SIZE
from doc_loader import extract_text_pdf_bytes
from services.exceptions import (
    DocumentParseError,
    FileTooLargeError,
    UnsupportedFileFormatError,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UploadResult:
    status: str
    length: int
    user_id: str


class FileService:
    def __init__(
        self, user_files: MutableMapping[str, str], max_file_size: int = MAX_FILE_SIZE
    ):
        self._user_files = user_files
        self._max_file_size = max_file_size

    def upload(
        self, filename: str | None, content_type: str | None, content: bytes
    ) -> UploadResult:
        if len(content) > self._max_file_size:
            raise FileTooLargeError("File too large, max size is 5 MB")

        text = self._extract_text(content_type, content)
        if not text or not text.strip():
            raise DocumentParseError("Could not extract text from the uploaded file")

        user_id = str(uuid.uuid4())
        self._user_files[user_id] = text

        logger.info("User %s uploaded file %s", user_id, filename)
        return UploadResult(status="ok", length=len(text), user_id=user_id)

    @staticmethod
    def _extract_text(content_type: str | None, content: bytes) -> str:
        if content_type == "application/pdf":
            text = extract_text_pdf_bytes(content)
            if text is None:
                raise DocumentParseError("Could not read PDF file")
            return text

        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnsupportedFileFormatError("Unsupported file format") from exc
