from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from indexer import Indexer
from models.schemas import ChunkMetadata, SearchResponse, SearchResult
from services.summarizer import SearchSummarizer
from .utils import get_indexer, get_summarizer, run_blocking

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Query text"),
    k: int = Query(10, description="Number of results"),
    summarize: bool = Query(True, description="Generate a natural language summary"),
    indexer: Indexer = Depends(get_indexer),
    summarizer: SearchSummarizer = Depends(get_summarizer),
):
    candidates = await run_blocking(indexer.search, q, k)
    if not candidates:
        fallback = "I couldn't find any information related to your request in the indexed data."
        return SearchResponse(query=q, results=[], summary=fallback)

    results = [
        SearchResult(
            chunk_id=c.chunk_id,
            score=float(c.score if c.score is not None else c.vector_score),
            metadata=ChunkMetadata(
                file_id=c.metadata.get("source_id", ""),
                file_path=c.metadata.get("source", ""),
                chunk_index=c.metadata.get("chunk_index", 0),
                start_time=c.metadata.get("start_time"),
                end_time=c.metadata.get("end_time"),
                file_type=c.metadata.get("file_type", "text"),
                text=c.metadata.get("text"),
            ),
        )
        for c in candidates
    ]
    summary = await run_blocking(summarizer.summarize, q, results) if summarize else None
    return SearchResponse(query=q, results=results, summary=summary)
