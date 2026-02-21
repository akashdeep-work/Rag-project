# RAG Backend API Request/Response Guide

This document is designed to be copied into AI prompts for frontend generation.

## Base Notes
- Base URL: `http://<host>:<port>`
- Auth model: **session scoped** using header `X-Session-Id`.
- Content type for JSON calls: `application/json`.
- Streaming endpoint uses **SSE** (`text/event-stream`).

---

## Common Header
```http
X-Session-Id: <client-generated-session-id>
```

Example:
```http
X-Session-Id: web_user_12345
```

---

## 1) Health
### `GET /health`
Checks API and index status.

**Response 200**
```json
{
  "status": "ok",
  "indexed_files": 2,
  "index_size": 540
}
```

---

## 2) Create Chat Session
### `POST /chat/sessions`
Create a new chat thread for the current `X-Session-Id`.

**Request**
```json
{
  "title": "Product Q&A"
}
```

**Response 201**
```json
{
  "id": 12,
  "title": "Product Q&A",
  "session_id": "web_user_12345"
}
```

---

## 3) List Chat Sessions
### `GET /chat/sessions`
List chat sessions for the current `X-Session-Id`.

**Response 200**
```json
[
  {
    "id": 12,
    "title": "Product Q&A",
    "created_at": "2026-02-20 10:10:10.123456"
  }
]
```

---

## 4) Chat History
### `GET /chat/{chat_id}/history`
Get all messages in a chat.

**Response 200**
```json
[
  {
    "id": 100,
    "role": "user",
    "content": "What is this file about?",
    "status": "completed",
    "timestamp": "2026-02-20 10:11:10.123456"
  },
  {
    "id": 101,
    "role": "assistant",
    "content": "This file explains...",
    "status": "completed",
    "timestamp": "2026-02-20 10:11:12.123456"
  }
]
```

---

## 5) Normal Chat API (non-streaming)
### `POST /chat/{chat_id}/message`
Sends a message and waits for full assistant response.

**Request**
```json
{
  "content": "Summarize the indexed documents"
}
```

**Response 200**
```json
{
  "assistant_message_id": 102,
  "role": "assistant",
  "status": "completed",
  "content": "Here is a concise summary..."
}
```

**Failure-style response example**
```json
{
  "assistant_message_id": 102,
  "role": "assistant",
  "status": "failed",
  "content": "Failed to generate response: <reason>"
}
```

---

## 6) Streaming Chat API (SSE)
### `POST /chat/{chat_id}/message/stream`
Streams assistant output token-by-token.

**Request**
```json
{
  "content": "Summarize the indexed documents"
}
```

**Response**
- HTTP 200
- `Content-Type: text/event-stream`
- Body is SSE events:

```text
data: {"assistant_message_id": 103, "status": "pending"}

data: {"token": "Here "}

data: {"token": "is "}

data: {"token": "the "}

data: {"token": "summary..."}

data: {"assistant_message_id": 103, "status": "completed"}
```

Frontend handling notes:
- On `pending`, create placeholder assistant message.
- Concatenate all `token` chunks into assistant text.
- On final `completed`, stop stream and keep persisted message id.

---

## 7) Single Message Status
### `GET /chat/messages/{message_id}`
Get current persisted state of one message.

**Response 200**
```json
{
  "id": 103,
  "role": "assistant",
  "status": "completed",
  "content": "Final assistant content...",
  "timestamp": "2026-02-20 10:13:12.123456"
}
```

---

## 8) Upload File
### `POST /upload`
Multipart upload + indexing.

**Request**
- `multipart/form-data`
- field: `file`

**Response 200**
```json
{
  "file_id": "abc123hash",
  "path": "data/guide.pdf",
  "indexed": true
}
```

---

## 9) List Uploaded Files
### `GET /upload/list`
List files for `X-Session-Id`.

**Response 200**
```json
[
  {
    "id": 1,
    "filename": "guide.pdf",
    "upload_date": "2026-02-20 09:10:10.123456"
  }
]
```

---

## 10) Search
### `GET /search?q=<query>&k=10&summarize=true`
Semantic search over indexed chunks.

**Response 200**
```json
{
  "query": "pricing",
  "results": [
    {
      "chunk_id": "<chunk-id>",
      "score": 0.83,
      "metadata": {
        "file_id": "<file-hash>",
        "file_path": "data/guide.pdf",
        "chunk_index": 2,
        "start_time": null,
        "end_time": null,
        "file_type": "text",
        "text": "..."
      }
    }
  ],
  "summary": "Optional LLM summary"
}
```

---

## 11) Get Chunk Detail
### `GET /chunk/{chunk_id}`
Returns metadata (+ vector if configured) for a chunk.

**Response 200**
```json
{
  "chunk_id": "<chunk-id>",
  "metadata": {
    "file_id": "<file-hash>",
    "file_path": "data/guide.pdf",
    "chunk_index": 2,
    "start_time": null,
    "end_time": null,
    "file_type": "text",
    "text": "..."
  },
  "vector": [0.01, -0.13, 0.44]
}
```

---

## Error Patterns
### Missing session header
**HTTP 400**
```json
{
  "detail": "X-Session-Id header is required"
}
```

### Not found (chat/message)
**HTTP 404**
```json
{
  "detail": "Session not found"
}
```
or
```json
{
  "detail": "Message not found"
}
```

