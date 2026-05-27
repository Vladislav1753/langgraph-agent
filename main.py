import logging
from contextlib import asynccontextmanager

from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import FastAPI

from agent import build_agent_graph
from routes import agent_requests, files
from services.agent_service import AgentRequestService
from services.file_service import FileService


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    app.state.agent = build_agent_graph().compile()
    app.state.user_files = TTLCache(maxsize=100, ttl=3600)
    app.state.file_service = FileService(app.state.user_files)
    app.state.agent_request_service = AgentRequestService(
        app.state.agent,
        app.state.user_files,
    )

    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(files.router, prefix="/files")
    app.include_router(agent_requests.router, prefix="/agent-request")
    return app


app = create_app()
