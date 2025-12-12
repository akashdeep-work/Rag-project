from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI

from config import HOST, PORT, DATA_DIR
from indexer import Indexer
from routers import search, upload, chunks, health,chat_router,auth_router
from services.background import BackgroundIndexer
from AiModel.models import LLM
from services.summarizer import SearchSummarizer
from db import engine, Base
from models import app_models
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that handles startup and shutdown logic.
    Everything before 'yield' runs on startup.
    Everything after 'yield' runs on shutdown.
    """
    # --- STARTUP LOGIC ---
    print("🚀 Server Starting Up...")
    
    # 1. Start Background Indexer
    if hasattr(app.state, "bg_indexer"):
        await app.state.bg_indexer.start()

    # 2. Check for Initial Indexing
    indexer_ref = app.state.indexer
    data_path = Path(DATA_DIR)
    
    # Check if registry is empty (implies no index) AND if we actually have files
    has_data = data_path.exists() and any(data_path.iterdir())
    is_index_empty = not indexer_ref._registry

    if is_index_empty and has_data:
        print(f"⚠️  No index found, but files detected in '{DATA_DIR}'.")
        print("⏳ Running initial indexing... (This may take a moment)")
        # Run indexing synchronously to ensure search is ready immediately
        indexer_ref.index_all()
        print("✅ Initial indexing complete.")
    elif is_index_empty:
        print(f"ℹ️  No index and no data found in '{DATA_DIR}'. Waiting for uploads.")
    else:
        print("✅ Existing index found. Ready.")

    # Yield control back to FastAPI to handle requests
    yield

    # --- SHUTDOWN LOGIC ---
    print("🛑 Server Shutting Down...")
    if hasattr(app.state, "bg_indexer"):
        await app.state.bg_indexer.stop()


def create_app() -> FastAPI:
    # Initialize Core Services
    indexer = Indexer()
    llm = LLM()
    summarizer = SearchSummarizer(llm)
    Base.metadata.create_all(bind=engine)
    # Pass the lifespan context manager here
    app = FastAPI(title="RAG Indexer API", version="1.0.0", lifespan=lifespan)

    origins = [
    "http://localhost:3000",     # React Dev
    "http://127.0.0.1:3000",
    "http://localhost:5173",     # Vite Dev
    "http://127.0.0.1:5173",
    # Add your production domain here:
    # "https://yourdomain.com"
]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store services in app state so they are accessible in lifespan
    app.state.indexer = indexer
    app.state.summarizer = summarizer
    # app.state.bg_indexer = BackgroundIndexer(indexer=indexer, interval_seconds=60)

    # Register Routers
    app.include_router(health.router)
    app.include_router(auth_router.router)
    app.include_router(upload.router)
    app.include_router(chat_router.router)
    app.include_router(search.router) # Your existing search router
    app.include_router(chunks.router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)