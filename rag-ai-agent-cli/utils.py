import hashlib
from typing import Any

def make_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf8")).hexdigest()

def chunk_id(source_id: str, chunk_index: int) -> str:
    return f"{source_id}::chunk::{chunk_index}"
