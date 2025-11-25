from __future__ import annotations
from fastapi import APIRouter, Depends

from indexer import Indexer
from models.schemas import HealthResponse
from .utils import get_indexer

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(indexer: Indexer = Depends(get_indexer)):
    indexed_files = len(indexer._registry)
    index_size = indexer.db.index.ntotal if indexer.db else 0
    return HealthResponse(indexed_files=indexed_files, index_size=index_size)
