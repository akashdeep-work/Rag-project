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
from models import app_models  # noqa: F401 - ensures models are registered
from routers import chat_router, chunks, health, search, upload
from routers.utils import run_blocking
from services.summarizer import SearchSummarizer

logger = logging.getLogger(__name__)

ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://karansharma.dev",
    "http://karansharma.dev",
    "https://akashdeep.cosmicowl.in",
    "http://akashdeep.cosmicowl.in",
)


def _configure_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Server starting up")
    indexer_ref: Indexer | None = getattr(app.state, "indexer", None)
    if indexer_ref is None:
        raise RuntimeError("Indexer service missing from application state.")

    await _run_initial_index(indexer_ref)
    yield
    logger.info("🛑 Server shutting down")


def create_app() -> FastAPI:
    _configure_logging()
    Base.metadata.create_all(bind=engine)

    app = FastAPI(title="RAG Indexer API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(ALLOWED_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.indexer = Indexer()
    app.state.summarizer = SearchSummarizer(LLM())

    app.include_router(health.router)
    app.include_router(upload.router)
    app.include_router(chat_router.router)
    app.include_router(search.router)
    app.include_router(chunks.router)
    return app


async def _run_initial_index(indexer_ref: Indexer) -> None:
    data_path = Path(DATA_DIR)
    has_data = data_path.exists() and any(data_path.iterdir())
    if indexer_ref._registry or not has_data:
        return

    logger.info("No index found. Running initial indexing in background worker...")
    await run_blocking(indexer_ref.index_all)
    logger.info("Initial indexing complete.")


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, timeout_keep_alive=300)
