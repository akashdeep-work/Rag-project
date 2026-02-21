from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    file_id: str = Field(..., description="Hashed identifier for the source file")
    file_path: str = Field(..., description="Original file path or URL")
    chunk_index: int
    start_time: Optional[float] = Field(None, description="Start time in seconds")
    end_time: Optional[float] = Field(None, description="End time in seconds")
    file_type: str = "text"
    text: Optional[str] = None


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    metadata: ChunkMetadata


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    summary: Optional[str] = Field(None, description="LLM generated summary of the retrieved context")


class UploadResponse(BaseModel):
    file_id: str
    path: str
    indexed: bool


class ChunkDetail(BaseModel):
    chunk_id: str
    metadata: ChunkMetadata
    vector: Optional[list[float]] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    indexed_files: int
    index_size: int
