from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from AiModel.models import LLM
from config import DATA_DIR, HOST, PORT
from db import Base, engine
from indexer import Indexer
from models import app_models
from routers import auth_router, chat_router, chunks, health, search, upload
from services.background import BackgroundIndexer
from services.summarizer import SearchSummarizer

logger = logging.getLogger(__name__)

ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def _configure_logging() -> None:
    """Configure application-wide logging for consistent output.

    Avoids re-configuration if handlers already exist to prevent duplicate logs
    when running under a process manager or test runner.
    """

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown concerns for the FastAPI application."""

    logger.info("🚀 Server starting up")

    # bg_indexer: BackgroundIndexer | None = getattr(app.state, "bg_indexer", None)
    # if bg_indexer:
    #     logger.info(
    #         "Starting background indexer with %s-second interval", bg_indexer.interval_seconds
    #     )
    #     await bg_indexer.start()

    indexer_ref: Indexer | None = getattr(app.state, "indexer", None)
    if indexer_ref is None:
        raise RuntimeError("Indexer service missing from application state.")

    _run_initial_index(indexer_ref)

    yield

    logger.info("🛑 Server shutting down")
    # if bg_indexer:
    #     await bg_indexer.stop()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    _configure_logging()
    _prepare_database()

    indexer, summarizer = _initialize_services()

    app = FastAPI(title="RAG Indexer API", version="1.0.0", lifespan=lifespan)

    _configure_cors(app)

    app.state.indexer = indexer
    app.state.summarizer = summarizer
    # app.state.bg_indexer = bg_indexer

    _register_routes(app)

    return app


def _configure_cors(app: FastAPI) -> None:
    """Attach CORS middleware with the supported origins."""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(ALLOWED_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _prepare_database() -> None:
    """Ensure database tables exist before the app starts serving requests."""

    Base.metadata.create_all(bind=engine)


def _initialize_services() -> tuple[Indexer, SearchSummarizer]:
    """Instantiate core services used by the FastAPI application."""

    indexer = Indexer()
    llm = LLM()
    summarizer = SearchSummarizer(llm)
    summarizer.summarize("test",[])
    # bg_indexer = BackgroundIndexer(indexer=indexer, interval_seconds=60)
    return indexer, summarizer


def _register_routes(app: FastAPI) -> None:
    """Register API routers with the FastAPI application."""

    app.include_router(health.router)
    app.include_router(auth_router.router)
    app.include_router(upload.router)
    app.include_router(chat_router.router)
    app.include_router(search.router)
    app.include_router(chunks.router)


def _run_initial_index(indexer_ref: Indexer) -> None:
    """Index existing documents on startup when needed.

    Ensures the application has search-ready data before serving requests. The
    process is synchronous because the API should not accept queries before the
    initial index is available.
    """

    data_path = Path(DATA_DIR)
    has_data = data_path.exists() and any(data_path.iterdir())
    is_index_empty = not indexer_ref._registry

    if not is_index_empty:
        logger.info("Existing index found. Ready for requests.")
        return

    if not has_data:
        logger.info("No index and no data found in '%s'. Waiting for uploads.", DATA_DIR)
        return

    logger.info(
        "No index found, but files detected in '%s'. Running initial indexing...",
        DATA_DIR,
    )
    try:
        indexer_ref.index_all()
    except Exception:  # pragma: no cover - fail fast during startup
        logger.exception("Initial indexing failed")
        raise
    logger.info("Initial indexing complete.")


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
