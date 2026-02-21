from __future__ import annotations

import json
from queue import Queue
from threading import Thread
from typing import Iterable, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from db import SessionLocal, get_db
from indexer import Indexer
from models.app_models import ChatMessage, ChatSession
from models.schemas import ChunkMetadata, SearchResult
from services.summarizer import SearchSummarizer
from .utils import get_indexer, get_session_id, get_summarizer, run_blocking

router = APIRouter(prefix="/chat", tags=["chat"])

CHIT_CHAT_CLUES = {"hi", "hello", "hey", "how are you", "what's up", "thank you", "thanks"}


class SessionCreate(BaseModel):
    title: str


class MessageCreate(BaseModel):
    content: str


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    status: str
    timestamp: str


def _to_results(candidates) -> list[SearchResult]:
    return [
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


def _persist_assistant_message(chat_id: int, assistant_message_id: int, text: str, status: str) -> None:
    db = SessionLocal()
    try:
        assistant_message = (
            db.query(ChatMessage)
            .filter(ChatMessage.id == assistant_message_id, ChatMessage.session_ref == chat_id)
            .first()
        )
        if assistant_message:
            assistant_message.content = text
            assistant_message.status = status
            db.commit()
    finally:
        db.close()


def _iter_model_tokens(
    user_text: str,
    indexer: Indexer,
    summarizer: SearchSummarizer,
) -> Iterable[str]:
    lower_content = user_text.strip().lower()
    looks_like_chitchat = any(clue in lower_content for clue in CHIT_CHAT_CLUES)
    candidates = [] if looks_like_chitchat else indexer.search(user_text, k=5)

    if not candidates:
        return summarizer.converse_stream(user_text)
    return summarizer.summarize_stream(user_text, _to_results(candidates))


def _generate_full_response_text(user_text: str, indexer: Indexer, summarizer: SearchSummarizer) -> str:
    return "".join(_iter_model_tokens(user_text, indexer, summarizer)).strip()


@router.post("/sessions", status_code=201)
async def create_chat_session(
    session_data: SessionCreate,
    db: Session = Depends(get_db),
    session_id: str = Depends(get_session_id),
):
    chat_session = ChatSession(title=session_data.title, session_id=session_id)
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return {"id": chat_session.id, "title": chat_session.title, "session_id": session_id}


@router.get("/sessions")
async def get_my_sessions(
    db: Session = Depends(get_db),
    session_id: str = Depends(get_session_id),
):
    sessions = db.query(ChatSession).filter(ChatSession.session_id == session_id).all()
    return [{"id": s.id, "title": s.title, "created_at": s.created_at} for s in sessions]


@router.get("/{chat_id}/history", response_model=List[MessageResponse])
async def get_chat_history(
    chat_id: int,
    db: Session = Depends(get_db),
    session_id: str = Depends(get_session_id),
):
    session = (
        db.query(ChatSession)
        .options(joinedload(ChatSession.messages))
        .filter(ChatSession.id == chat_id, ChatSession.session_id == session_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "status": m.status,
            "timestamp": str(m.timestamp),
        }
        for m in session.messages
    ]


@router.post("/{chat_id}/message/stream")
async def stream_message(
    chat_id: int,
    message: MessageCreate,
    db: Session = Depends(get_db),
    session_id: str = Depends(get_session_id),
    indexer: Indexer = Depends(get_indexer),
    summarizer: SearchSummarizer = Depends(get_summarizer),
):
    session = db.query(ChatSession).filter(ChatSession.id == chat_id, ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    db.add(ChatMessage(session_ref=session.id, role="user", content=message.content, status="completed"))
    assistant = ChatMessage(session_ref=session.id, role="assistant", content="", status="pending")
    db.add(assistant)
    db.commit()
    db.refresh(assistant)

    stream_queue: Queue[str | None] = Queue()

    def _producer() -> None:
        buffer: list[str] = []
        try:
            for token in _iter_model_tokens(message.content, indexer, summarizer):
                buffer.append(token)
                stream_queue.put(token)

            final_text = "".join(buffer).strip()
            _persist_assistant_message(session.id, assistant.id, final_text, "completed")
        except Exception as exc:  # pragma: no cover
            error_text = f"Failed to generate response: {exc}"
            _persist_assistant_message(session.id, assistant.id, error_text, "failed")
            stream_queue.put(f"\n[ERROR] {error_text}")
        finally:
            stream_queue.put(None)

    Thread(target=_producer, daemon=True).start()

    async def event_stream():
        yield f"data: {json.dumps({'assistant_message_id': assistant.id, 'status': 'pending'})}\n\n"

        while True:
            chunk = await run_blocking(stream_queue.get)
            if chunk is None:
                yield f"data: {json.dumps({'assistant_message_id': assistant.id, 'status': 'completed'})}\n\n"
                break
            yield f"data: {json.dumps({'token': chunk})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/messages/{message_id}")
async def get_message_status(
    message_id: int,
    db: Session = Depends(get_db),
    session_id: str = Depends(get_session_id),
):
    message = (
        db.query(ChatMessage)
        .join(ChatSession, ChatMessage.session_ref == ChatSession.id)
        .filter(ChatMessage.id == message_id, ChatSession.session_id == session_id)
        .first()
    )
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    return {
        "id": message.id,
        "role": message.role,
        "status": message.status,
        "content": message.content,
        "timestamp": str(message.timestamp),
    }


@router.post("/{chat_id}/message")
async def send_message(
    chat_id: int,
    message: MessageCreate,
    db: Session = Depends(get_db),
    session_id: str = Depends(get_session_id),
    indexer: Indexer = Depends(get_indexer),
    summarizer: SearchSummarizer = Depends(get_summarizer),
):
    session = db.query(ChatSession).filter(ChatSession.id == chat_id, ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_msg = ChatMessage(session_ref=session.id, role="user", content=message.content, status="completed")
    db.add(user_msg)
    db.commit()

    assistant = ChatMessage(session_ref=session.id, role="assistant", content="", status="pending")
    db.add(assistant)
    db.commit()
    db.refresh(assistant)

    try:
        reply_text = await run_blocking(_generate_full_response_text, message.content, indexer, summarizer)
        assistant.content = reply_text
        assistant.status = "completed"
    except Exception as exc:  # pragma: no cover
        assistant.content = f"Failed to generate response: {exc}"
        assistant.status = "failed"

    db.commit()

    return {
        "assistant_message_id": assistant.id,
        "role": "assistant",
        "status": assistant.status,
        "content": assistant.content,
    }
