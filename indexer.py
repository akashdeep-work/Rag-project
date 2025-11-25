from __future__ import annotations
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from config import (
    DATA_DIR,
    EMBED_BATCH,
    INDEX_ADD_BATCH,
    EXTRACTION_THREADS,
    ANN_CANDIDATES,
    RERANK_TOPK,
    STORE_DIR,
)
from reader import (
    read_pdf,
    read_docx,
    read_txt,
    read_url,
    chunk_text,
    transcribe_media,
    chunk_transcript,
    TranscriptSegment,
)
from AiModel.models import Embedder
from utils import make_hash, chunk_id
from FaissDB.FaissHNSWDB import FaissHNSWDB
from services.rerank import LinearReranker, SearchCandidate

# Prevent TQDM warnings in threads
tqdm.monitor_interval = 0

SUPPORTED_TEXT_SUFFIXES = {".pdf", ".docx", ".txt"}
SUPPORTED_MEDIA_SUFFIXES = {".mp3", ".wav", ".mp4", ".mkv"}


@dataclass
class ChunkMetadata:
    """Metadata stored alongside each chunk."""

    text: str
    source: str
    chunk_index: int
    source_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    file_type: str = "text"
    modified_at: Optional[float] = None

    def to_db(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "source_id": self.source_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "file_type": self.file_type,
            "modified_at": self.modified_at,
        }


@dataclass
class IndexedFile:
    source_id: str
    path: str
    modified_at: float
    file_type: str


class Indexer:
    def __init__(
        self,
        db_store_dir: str = "rag_store",
        embed_model: str = "all-MiniLM-L6-v2",
        registry_path: Optional[Path] = None,
    ):
        self.db_dir = db_store_dir
        self.registry_path = registry_path or STORE_DIR / "indexed_files.json"

        # Initialize Embedder
        print(f"Loading embedder: {embed_model}...")
        self.embedder = Embedder(embed_model)
        self.dim = self.embedder.dim

        # Initialize Vector DB
        print(f"Initializing FaissHNSWDB in {self.db_dir}...")
        self.db = FaissHNSWDB(
            dim=self.dim,
            path=self.db_dir,
            M=32,
            ef_construction=200,
            ef_search=50,
            metric="cosine",
        )

        self.reranker = LinearReranker()
        self._registry = self._load_registry()

    # ------------------------------------------------------------------
    # Source enumeration and extraction
    # ------------------------------------------------------------------
    def _load_registry(self) -> Dict[str, IndexedFile]:
        if Path(self.registry_path).exists():
            try:
                data = json.loads(Path(self.registry_path).read_text())
                return {
                    k: IndexedFile(**v)
                    for k, v in data.items()
                }
            except Exception:
                return {}
        return {}

    def _save_registry(self) -> None:
        serializable = {k: vars(v) for k, v in self._registry.items()}
        Path(self.registry_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.registry_path).write_text(json.dumps(serializable, indent=2))

    def _list_sources(self) -> List[str]:
        """Scans DATA_DIR for valid files and urls.txt."""
        files: List[str] = []
        if not os.path.exists(DATA_DIR):
            print(f"Warning: {DATA_DIR} does not exist.")
            return files

        p_dir = Path(DATA_DIR)

        for p in p_dir.rglob("*"):
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            if suffix in SUPPORTED_TEXT_SUFFIXES | SUPPORTED_MEDIA_SUFFIXES:
                files.append(str(p.resolve()))

        urls_file = p_dir / "urls.txt"
        if urls_file.exists():
            try:
                for line in urls_file.read_text("utf-8").splitlines():
                    u = line.strip()
                    if u and (u.startswith("http://") or u.startswith("https://")):
                        files.append(u)
            except Exception as e:
                print(f"Error reading urls.txt: {e}")

        return files

    def _extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return read_pdf(path)
        if suffix == ".docx":
            return read_docx(path)
        if suffix == ".txt":
            return read_txt(path)
        return ""

    def _extract(self, source: str) -> Tuple[Optional[str], Optional[List[TranscriptSegment]], Optional[str]]:
        """Reads content from a file path or URL and returns transcript segments.

        Returns: (source_hash_id, transcript_segments, file_type)
        """
        try:
            if source.startswith("http://") or source.startswith("https://"):
                text = read_url(source)
                src_id = make_hash(source)
                segment = TranscriptSegment(text=text, start=0.0, end=0.0)
                return src_id, [segment], "url"

            p = Path(source)
            suffix = p.suffix.lower()

            if suffix in SUPPORTED_TEXT_SUFFIXES:
                text = self._extract_text(p)
                segment = TranscriptSegment(text=text, start=0.0, end=0.0)
                src_id = make_hash(str(p.resolve()))
                return src_id, [segment], "text"

            if suffix in SUPPORTED_MEDIA_SUFFIXES:
                segments = transcribe_media(p)
                src_id = make_hash(str(p.resolve()))
                return src_id, segments, "media"

            return None, None, None
        except Exception as e:
            print(f"Failed to extract {source}: {e}")
            return None, None, None

    def _chunk_generator(
        self, sources: List[str], chunk_size: int, chunk_overlap: int
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield chunks one by one to avoid holding full text in memory."""
        with ThreadPoolExecutor(max_workers=EXTRACTION_THREADS) as ex:
            future_to_src = {ex.submit(self._extract, s): s for s in sources}

            for future in as_completed(future_to_src):
                src = future_to_src[future]
                try:
                    sid, segments, file_type = future.result()
                    if not segments:
                        continue

                    if file_type == "media":
                        chunked_segments = chunk_transcript(
                            segments, chunk_size=chunk_size, overlap=chunk_overlap
                        )
                    else:
                        all_text = "\n".join(seg.text for seg in segments if seg.text)
                        text_chunks = chunk_text(all_text, chunk_size, chunk_overlap)
                        chunked_segments = [
                            TranscriptSegment(text=t, start=None, end=None)
                            for t in text_chunks
                        ]

                    for i, seg in enumerate(chunked_segments):
                        cid = chunk_id(sid, i)
                        yield {
                            "id": cid,
                            "text": seg.text,
                            "source_id": sid,
                            "source_name": src,
                            "chunk_index": i,
                            "start_time": seg.start,
                            "end_time": seg.end,
                            "file_type": file_type,
                        }
                except Exception as e:
                    print(f"Error processing {src}: {e}")

    # ------------------------------------------------------------------
    # Indexing and persistence
    # ------------------------------------------------------------------
    def _should_index(self, path: Path, source_id: str) -> bool:
        """Check if a file should be indexed based on modification time."""
        if source_id in self._registry:
            known = self._registry[source_id]
            try:
                if path.stat().st_mtime <= known.modified_at:
                    return False
            except FileNotFoundError:
                return False
        return True

    def _process_batch(self, ids: List[str], texts: List[str], metas: List[dict]):
        try:
            embeddings = self.embedder.embed_texts(texts, batch_size=EMBED_BATCH)
            self.db.add(user_ids=ids, vectors=embeddings, metadatas=metas)
        except Exception as e:
            print(f"Batch processing failed: {e}")

    def _update_registry(self, path: Path, source_id: str, file_type: str) -> None:
        try:
            modified = path.stat().st_mtime
        except FileNotFoundError:
            modified = 0.0
        self._registry[source_id] = IndexedFile(
            source_id=source_id,
            path=str(path),
            modified_at=modified,
            file_type=file_type,
        )
        self._save_registry()

    def index_sources(self, sources: Iterable[str], chunk_size: int = 500, chunk_overlap: int = 50):
        sources = list(sources)
        print(f"Found {len(sources)} sources to process.")
        if not sources:
            return

        chunk_gen = self._chunk_generator(sources, chunk_size, chunk_overlap)

        current_batch_texts: List[str] = []
        current_batch_ids: List[str] = []
        current_batch_metas: List[dict] = []

        total_added = 0
        pbar = tqdm(desc="Indexing Chunks", unit="chunk")

        for item in chunk_gen:
            current_batch_ids.append(item["id"])
            current_batch_texts.append(item["text"])

            meta = ChunkMetadata(
                text=item["text"],
                source=item["source_name"],
                chunk_index=item["chunk_index"],
                source_id=item["source_id"],
                start_time=item.get("start_time"),
                end_time=item.get("end_time"),
                file_type=item.get("file_type", "text"),
                modified_at=Path(item["source_name"]).stat().st_mtime
                if Path(item["source_name"]).exists()
                else None,
            )
            current_batch_metas.append(meta.to_db())

            if len(current_batch_texts) >= INDEX_ADD_BATCH:
                self._process_batch(current_batch_ids, current_batch_texts, current_batch_metas)
                total_added += len(current_batch_texts)
                pbar.update(len(current_batch_texts))

                current_batch_ids = []
                current_batch_texts = []
                current_batch_metas = []

                if total_added % (INDEX_ADD_BATCH * 5) == 0:
                    gc.collect()

        if current_batch_texts:
            self._process_batch(current_batch_ids, current_batch_texts, current_batch_metas)
            total_added += len(current_batch_texts)
            pbar.update(len(current_batch_texts))

        pbar.close()
        print(f"Indexing complete. Total chunks added: {total_added}")
        self.db.save()

        for src in sources:
            if src.startswith("http"):
                continue
            source_id = make_hash(str(Path(src).resolve()))
            suffix = Path(src).suffix.lower()
            file_type = "media" if suffix in SUPPORTED_MEDIA_SUFFIXES else "text"
            self._update_registry(Path(src), source_id, file_type)

    def index_all(self, chunk_size: int = 500, chunk_overlap: int = 50):
        sources = self._list_sources()
        self.index_sources(sources, chunk_size, chunk_overlap)

    def index_file(self, file_path: Path, chunk_size: int = 500, chunk_overlap: int = 50):
        """Index a single file path if it is new or modified."""
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_TEXT_SUFFIXES | SUPPORTED_MEDIA_SUFFIXES:
            print(f"Skipping unsupported file type: {file_path}")
            return

        source_id = make_hash(str(file_path.resolve()))
        if not self._should_index(file_path, source_id):
            print(f"File already indexed and up-to-date: {file_path}")
            return

        self.index_sources([str(file_path)], chunk_size, chunk_overlap)

    def index_new_files(self, chunk_size: int = 500, chunk_overlap: int = 50):
        sources = self._list_sources()
        to_index: List[str] = []
        for src in sources:
            if src.startswith("http"):
                continue
            p = Path(src)
            source_id = make_hash(str(p.resolve()))
            if self._should_index(p, source_id):
                to_index.append(src)
        if to_index:
            self.index_sources(to_index, chunk_size, chunk_overlap)

    # ------------------------------------------------------------------
    # Search and retrieval
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 10) -> List[SearchCandidate]:
        q_emb = self.embedder.embed(query)
        if not isinstance(q_emb, np.ndarray):
            q_emb = np.array(q_emb)

        print(f"Searching for: '{query}' (k={k})")
        initial_k = max(k, ANN_CANDIDATES)
        results = self.db.search(q_emb, k=initial_k)
        candidates = [
            SearchCandidate(chunk_id=uid, vector_score=score, metadata=meta or {})
            for uid, score, meta in results
        ]
        reranked = self.reranker.rerank(query, candidates, top_k=max(k, RERANK_TOPK))
        return reranked[:k]

    def get_chunk(self, chunk_id_value: str) -> Optional[Dict[str, Any]]:
        record = self.db.get(chunk_id_value)
        if not record:
            return None
        meta = record.get("metadata") or {}
        return {
            "chunk_id": chunk_id_value,
            "metadata": meta,
            "vector": record.get("vector"),
        }


if __name__ == "__main__":
    indexer = Indexer()
    indexer.index_all()
    results = indexer.search("What is the financial outlook?", k=3)
    for res in results:
        print(f"\n[Score: {res.score:.4f}] {res.chunk_id}")
        text_snippet = res.metadata.get("text", "")
        print(f"Text snippet: {text_snippet[:100]}...")
