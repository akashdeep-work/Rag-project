from __future__ import annotations
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException

from config import DATA_DIR
from indexer import Indexer
from models.schemas import UploadResponse
from .utils import get_indexer
from utils import make_hash

router = APIRouter(prefix="/upload", tags=["ingestion"])


def _save_upload(file: UploadFile) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / file.filename
    with destination.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return destination


@router.post("", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    indexer: Indexer = Depends(get_indexer),
):
    destination = _save_upload(file)
    file_id = make_hash(str(destination.resolve()))

    try:
        indexer.index_file(destination)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to index file: {exc}")

    return UploadResponse(file_id=file_id, path=str(destination), indexed=True)
