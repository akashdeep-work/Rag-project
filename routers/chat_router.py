# routers/chat.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List

from db import get_db
from middleware.auth import get_current_user, AuthUser
from models.app_models import User, ChatSession, ChatMessage
from indexer import Indexer
from services.summarizer import SearchSummarizer
from models.schemas import SearchResult, ChunkMetadata
from .utils import get_indexer, get_summarizer

router = APIRouter(prefix="/chat", tags=["chat"])

# --- Schemas ---
class SessionCreate(BaseModel):
    title: str

class MessageCreate(BaseModel):
    content: str

class MessageResponse(BaseModel):
    role: str
    content: str
    timestamp: str

# --- Endpoints ---

@router.post("/sessions", status_code=201)
def create_chat_session(
    session_data: SessionCreate,
    db: Session = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user)
):
    """Create a new chat session."""
    new_session = ChatSession(title=session_data.title, user_id=current_user.id)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return {"id": new_session.id, "title": new_session.title, "token":current_user.token}

@router.get("/sessions")
def get_my_sessions(
    db: Session = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user)
):
    """List all chat sessions for the logged-in user."""
    sessions = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).all()
    return [{"id": s.id, "title": s.title, "created_at": s.created_at} for s in sessions]

@router.get("/{session_id}/history", response_model=List[MessageResponse])
def get_chat_history(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user)
):
    """Get message history for a specific session."""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id, 
        # ChatSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return [
        {"role": m.role, "content": m.content, "timestamp": str(m.timestamp)} 
        for m in session.messages
    ]

@router.post("/{session_id}/message")
def send_message(
    session_id: int,
    message: MessageCreate,
    db: Session = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
    indexer: Indexer = Depends(get_indexer),
    summarizer: SearchSummarizer = Depends(get_summarizer)
):
    """Send a message, get RAG response, and save both to history."""
    print(f'user data id is :{current_user.id}')
    # 1. Validate Session
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id, 
        ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 2. Save User Message
    user_msg = ChatMessage(session_id=session.id, role="user", content=message.content)
    db.add(user_msg)
    db.commit()

    # 3. Perform RAG Search (Logic reused from your search API)
    candidates = indexer.search(message.content, k=5)
    results = [
        SearchResult(
            chunk_id=c.chunk_id,
            score=float(c.score if c.score is not None else c.vector_score),
            metadata=ChunkMetadata(
                file_id=c.metadata.get("source_id", ""),
                file_path=c.metadata.get("source", ""),
                chunk_index=c.metadata.get("chunk_index", 0),
                file_type=c.metadata.get("file_type", "text"),
                text=c.metadata.get("text"),
            ),
        )
        for c in candidates
    ]

    # 4. Generate AI Response
    ai_response_text = summarizer.summarize(message.content, results)

    # 5. Save AI Message
    bot_msg = ChatMessage(session_id=session.id, role="assistant", content=ai_response_text)
    db.add(bot_msg)
    db.commit()

    return {"role": "assistant", "content": ai_response_text}