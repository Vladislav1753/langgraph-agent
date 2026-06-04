import logging
from contextlib import asynccontextmanager

from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import FastAPI
from langfuse import get_client

from app.agent.agent import build_agent_graph
from app.agent.tools import set_document_store
from app.core.config import settings
from app.core.handlers import setup_exception_handlers
from app.routers import agent_requests, files, tasks
from app.services.agent import AgentRequestService
from app.services.file import FileService
from app.services.task_service import TaskService
from app.services.task_store import InMemoryTaskStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    app.state.agent = build_agent_graph().compile()
    app.state.user_files = TTLCache(maxsize=100, ttl=3600)
    set_document_store(app.state.user_files)
    app.state.file_service = FileService(app.state.user_files)
    app.state.agent_request_service = AgentRequestService(
        app.state.agent,
        app.state.user_files,
    )
    app.state.task_store = InMemoryTaskStore(
        maxsize=settings.task_store_maxsize, ttl=settings.task_store_ttl
    )
    app.state.task_service = TaskService(
        agent_service=app.state.agent_request_service, store=app.state.task_store
    )

    logger.info("Application startup complete")
    try:
        yield
    finally:
        get_client().flush()
        logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    setup_exception_handlers(app)

    app.include_router(files.router, prefix="/files")
    app.include_router(tasks.router, prefix="/tasks")
    app.include_router(agent_requests.router, prefix="/agent-request")
    return app


app = create_app()
