from __future__ import annotations
import asyncio
from fastapi import FastAPI

from config import HOST, PORT
from indexer import Indexer
from routers import search, upload, chunks, health
from services.background import BackgroundIndexer
from AiModel.models import LLM
from services.summarizer import SearchSummarizer


def create_app() -> FastAPI:
    indexer = Indexer()
    llm = LLM()
    summarizer = SearchSummarizer(llm)
    app = FastAPI(title="RAG Indexer API", version="1.0.0")
    app.state.indexer = indexer
    app.state.summarizer = summarizer
    app.state.bg_indexer = BackgroundIndexer(indexer=indexer, interval_seconds=60)

    app.include_router(search.router)
    app.include_router(upload.router)
    app.include_router(chunks.router)
    app.include_router(health.router)

    @app.on_event("startup")
    async def startup_event() -> None:
        await app.state.bg_indexer.start()

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        await app.state.bg_indexer.stop()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
