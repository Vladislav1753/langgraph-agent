import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from http import HTTPStatus

from app.core.exceptions import (
    FileTooLargeError,
    UnsupportedFileFormatError,
    DocumentParseError,
    ObjectNotFoundError,
    AgentExecutionError,
)

logger = logging.getLogger(__name__)


def setup_exception_handlers(app: FastAPI):
    @app.exception_handler(ObjectNotFoundError)
    async def object_not_found_handler(request: Request, exc: ObjectNotFoundError):
        logger.error(str(exc))
        return JSONResponse(
            status_code=HTTPStatus.NOT_FOUND,
            content={"detail": exc.message},
        )

    @app.exception_handler(FileTooLargeError)
    async def file_too_large_handler(request: Request, exc: FileTooLargeError):
        logger.error(str(exc))
        return JSONResponse(
            status_code=HTTPStatus.CONTENT_TOO_LARGE,
            content={"detail": exc.message},
        )

    @app.exception_handler(UnsupportedFileFormatError)
    async def unsupported_format_handler(
        request: Request, exc: UnsupportedFileFormatError
    ):
        logger.error(str(exc))
        return JSONResponse(
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            content={"detail": exc.message},
        )

    @app.exception_handler(DocumentParseError)
    async def parser_error_handler(request: Request, exc: DocumentParseError):
        logger.error(str(exc))
        return JSONResponse(
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            content={"detail": exc.message},
        )

    @app.exception_handler(AgentExecutionError)
    async def conflict_error_handler(request: Request, exc: AgentExecutionError):
        logger.error(str(exc))

        return JSONResponse(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            content={"detail": exc.message},
        )
