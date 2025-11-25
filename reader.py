from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from PyPDF2 import PdfReader
from docx import Document
import requests
from bs4 import BeautifulSoup


@dataclass
class TranscriptSegment:
    """Represents a single segment of transcribed audio or video."""

    text: str
    start: float
    end: float


# simple util: split into chunks by sentences/words ~ target_tokens approximation
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks


def read_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        text = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                text.append(t)
        return "\n".join(text)
    except Exception:
        return ""


def read_docx(path: Path) -> str:
    try:
        d = Document(str(path))
        return "\n".join([p.text for p in d.paragraphs])
    except Exception:
        return ""


def read_txt(path: Path) -> str:
    try:
        return Path(path).read_text(errors="ignore")
    except Exception:
        return ""


def read_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return soup.get_text(separator="\n")
    except Exception:
        return ""


def transcribe_media(path: Path) -> List[TranscriptSegment]:
    """
    Stub for media transcription.

    In production this function should call a speech-to-text engine and return
    timestamped segments. The stub keeps the interface stable for downstream
    logic while making local development deterministic.
    """
    placeholder = f"Transcribed content for {path.name}."
    # Assume a 60 second clip with a single segment for simplicity
    return [TranscriptSegment(text=placeholder, start=0.0, end=60.0)]


def chunk_transcript(
    segments: Iterable[TranscriptSegment], chunk_size: int, overlap: int
) -> List[TranscriptSegment]:
    """
    Chunk transcript segments while preserving time boundaries.

    The algorithm splits text by words similar to ``chunk_text`` but maps
    proportional timestamps into each chunk based on the original segment
    duration. This keeps a stable contract for media snippets returned to
    clients.
    """
    chunked: List[TranscriptSegment] = []
    for segment in segments:
        words = segment.text.split()
        if not words:
            continue
        duration = max(segment.end - segment.start, 0.001)
        total_words = len(words)
        idx = 0
        while idx < total_words:
            window = words[idx : idx + chunk_size]
            rel_start = idx / total_words
            rel_end = min(total_words, idx + chunk_size) / total_words
            start_time = segment.start + duration * rel_start
            end_time = segment.start + duration * rel_end
            chunked.append(
                TranscriptSegment(
                    text=" ".join(window), start=start_time, end=end_time
                )
            )
            idx += max(chunk_size - overlap, 1)
    return chunked
