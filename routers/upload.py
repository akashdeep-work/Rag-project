# routers/upload.py
from __future__ import annotations
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session

from config import DATA_DIR
from indexer import Indexer
from models.schemas import UploadResponse
from models.app_models import UploadedFile, User
from db import get_db
from middleware.auth import get_current_user
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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # Requires Login
):
    destination = _save_upload(file)
    file_id = make_hash(str(destination.resolve()))

    try:
        indexer.index_file(destination)
        
        # Save to SQLite
        new_file = UploadedFile(
            filename=file.filename,
            file_path=str(destination),
            file_hash=file_id,
            user_id=current_user.id
        )
        db.add(new_file)
        db.commit()
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to index file: {exc}")

    return UploadResponse(file_id=file_id, path=str(destination), indexed=True)

@router.get("/list", tags=["ingestion"])
def list_my_files(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get list of uploaded files for the current user."""
    files = db.query(UploadedFile).filter(UploadedFile.user_id == current_user.id).all()
    return [
        {"id": f.id, "filename": f.filename, "upload_date": f.upload_date} 
        for f in files
    ]