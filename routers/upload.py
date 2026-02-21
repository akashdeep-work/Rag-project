from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from config import DATA_DIR
from db import get_db
from indexer import Indexer
from models.app_models import UploadedFile
from models.schemas import UploadResponse
from utils import make_hash
from .utils import get_indexer, get_session_id, run_blocking

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
    db: Session = Depends(get_db),
    session_id: str = Depends(get_session_id),
):
    destination = _save_upload(file)
    file_id = make_hash(str(destination.resolve()))

    try:
        await run_blocking(indexer.index_file, destination)

        new_file = UploadedFile(
            filename=file.filename,
            file_path=str(destination),
            file_hash=file_id,
            session_id=session_id,
        )
        db.add(new_file)
        db.commit()

    except Exception as exc:  # pragma: no cover - startup path
        raise HTTPException(status_code=500, detail=f"Failed to index file: {exc}")

    return UploadResponse(file_id=file_id, path=str(destination), indexed=True)


@router.get("/list", tags=["ingestion"])
async def list_my_files(
    db: Session = Depends(get_db),
    session_id: str = Depends(get_session_id),
):
    files = db.query(UploadedFile).filter(UploadedFile.session_id == session_id).all()
    return [{"id": f.id, "filename": f.filename, "upload_date": f.upload_date} for f in files]
