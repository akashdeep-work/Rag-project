from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, TypeVar

from fastapi import Header, HTTPException, Request

from indexer import Indexer
from services.summarizer import SearchSummarizer

T = TypeVar("T")

# Shared thread-pool for blocking work (indexing, vector search, LLM calls, sync DB work).
_executor = ThreadPoolExecutor(max_workers=16)


def get_indexer(request: Request) -> Indexer:
    return request.app.state.indexer


def get_summarizer(request: Request) -> SearchSummarizer:
    return request.app.state.summarizer


def get_session_id(x_session_id: str | None = Header(default=None)) -> str:
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-Id header is required")
    return x_session_id.strip()


async def run_blocking(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    call = partial(func, *args, **kwargs)
    return await loop.run_in_executor(_executor, call)
