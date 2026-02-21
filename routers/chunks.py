from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from indexer import Indexer
from models.schemas import ChunkDetail, ChunkMetadata
from .utils import get_indexer, run_blocking

router = APIRouter(prefix="/chunk", tags=["chunks"])


@router.get("/{chunk_id}", response_model=ChunkDetail)
async def get_chunk(chunk_id: str, indexer: Indexer = Depends(get_indexer)):
    record = await run_blocking(indexer.get_chunk, chunk_id)
    if not record:
        raise HTTPException(status_code=404, detail="Chunk not found")
    meta = record.get("metadata", {})
    metadata = ChunkMetadata(
        file_id=meta.get("source_id", ""),
        file_path=meta.get("source", ""),
        chunk_index=meta.get("chunk_index", 0),
        start_time=meta.get("start_time"),
        end_time=meta.get("end_time"),
        file_type=meta.get("file_type", "text"),
        text=meta.get("text"),
    )
    return ChunkDetail(chunk_id=chunk_id, metadata=metadata, vector=record.get("vector"))
