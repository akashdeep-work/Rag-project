from pathlib import Path
from typing import List
import re
from PyPDF2 import PdfReader
from docx import Document
import requests
from bs4 import BeautifulSoup
import math

# simple util: split into chunks by sentences/words ~ target_tokens approximation
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    # approximate tokens by words: chunk_size ~ tokens
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
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
