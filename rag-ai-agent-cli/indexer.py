import os
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Generator, Dict, Any, Optional
import numpy as np
from tqdm import tqdm

# Import the CrossEncoder for re-ranking
from sentence_transformers import CrossEncoder

from config import (
    DATA_DIR, 
    EMBED_BATCH, 
    INDEX_ADD_BATCH, 
    EXTRACTION_THREADS
)
from reader import read_pdf, read_docx, read_txt, read_url, chunk_text
from AiModel.models import Embedder
from utils import make_hash, chunk_id
from FaissDB.FaissHNSWDB import FaissHNSWDB  

# Prevent TQDM warnings in threads
tqdm.monitor_interval = 0

class Indexer:
    def __init__(
        self, 
        db_store_dir: str = "rag_store", 
        embed_model: str = "all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2" # <--- NEW PARAM
    ):
        self.db_dir = db_store_dir
        
        # 1. Initialize Bi-Encoder (Vector Embedder)
        print(f"Loading embedder: {embed_model}...")
        self.embedder = Embedder(embed_model)
        self.dim = self.embedder.dim

        # 2. Initialize Cross-Encoder (Re-ranker)
        # We load this lazily or immediately. For simplicity, we load it here.
        # This runs on CPU by default if no GPU is found, which is fine for re-ranking small batches.
        print(f"Loading re-ranker: {rerank_model}...")
        try:
            self.reranker = CrossEncoder(rerank_model)
        except Exception as e:
            print(f"Warning: Could not load re-ranker {rerank_model}. Re-ranking will be disabled. Error: {e}")
            self.reranker = None
        
        # 3. Initialize Vector DB
        print(f"Initializing FaissHNSWDB in {self.db_dir}...")
        self.db = FaissHNSWDB(
            dim=self.dim, 
            path=self.db_dir, 
            M=32, 
            ef_construction=200, 
            ef_search=50, 
            metric="cosine"
        )

    # ... [Keep _list_sources, _extract, _chunk_generator, index_all, _process_batch exactly as they were] ...
    # (For brevity, I am not repeating the unchanged methods here. Assume they exist.)
    
    def _list_sources(self) -> List[str]:
        """Scans DATA_DIR for valid files and urls.txt."""
        files = []
        if not os.path.exists(DATA_DIR):
            print(f"Warning: {DATA_DIR} does not exist.")
            return files
        p_dir = Path(DATA_DIR)
        for p in p_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".pdf", ".docx", ".txt"]:
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

    def _extract(self, source: str) -> Tuple[str, str]:
        try:
            if source.startswith("http://") or source.startswith("https://"):
                text = read_url(source)
                src_id = make_hash(source)
                return src_id, text
            p = Path(source)
            suffix = p.suffix.lower()
            if suffix == ".pdf": text = read_pdf(p)
            elif suffix == ".docx": text = read_docx(p)
            elif suffix == ".txt": text = read_txt(p)
            else: return None, None
            src_id = make_hash(str(p.resolve()))
            return src_id, text
        except Exception as e:
            print(f"Failed to extract {source}: {e}")
            return None, None

    def _chunk_generator(self, sources: List[str], chunk_size: int, chunk_overlap: int) -> Generator[Dict[str, Any], None, None]:
        with ThreadPoolExecutor(max_workers=EXTRACTION_THREADS) as ex:
            future_to_src = {ex.submit(self._extract, s): s for s in sources}
            for future in as_completed(future_to_src):
                src = future_to_src[future]
                try:
                    sid, text = future.result()
                    if not text or not text.strip(): continue
                    chunks = chunk_text(text, chunk_size, chunk_overlap)
                    for i, c in enumerate(chunks):
                        cid = chunk_id(sid, i)
                        yield {"id": cid, "text": c, "source_id": sid, "source_name": src, "chunk_index": i}
                except Exception as e:
                    print(f"Error processing {src}: {e}")

    def index_all(self, chunk_size: int = 500, chunk_overlap: int = 50):
        sources = self._list_sources()
        print(f"Found {len(sources)} sources to process.")
        if not sources: return
        chunk_gen = self._chunk_generator(sources, chunk_size, chunk_overlap)
        current_batch_texts = []
        current_batch_ids = []
        current_batch_metas = []
        total_added = 0
        pbar = tqdm(desc="Indexing Chunks", unit="chunk")
        for item in chunk_gen:
            current_batch_ids.append(item['id'])
            current_batch_texts.append(item['text'])
            meta = {"text": item['text'], "source": item['source_name'], "chunk_index": item['chunk_index']}
            current_batch_metas.append(meta)
            if len(current_batch_texts) >= INDEX_ADD_BATCH:
                self._process_batch(current_batch_ids, current_batch_texts, current_batch_metas)
                total_added += len(current_batch_texts)
                pbar.update(len(current_batch_texts))
                current_batch_ids = []
                current_batch_texts = []
                current_batch_metas = []
                if total_added % (INDEX_ADD_BATCH * 5) == 0: gc.collect()
        if current_batch_texts:
            self._process_batch(current_batch_ids, current_batch_texts, current_batch_metas)
            total_added += len(current_batch_texts)
            pbar.update(len(current_batch_texts))
        pbar.close()
        print(f"Indexing complete. Total chunks added: {total_added}")
        self.db.save()

    def _process_batch(self, ids: List[str], texts: List[str], metas: List[dict]):
        try:
            embeddings = self.embedder.embed_texts(texts, batch_size=EMBED_BATCH)
            self.db.add(user_ids=ids, vectors=embeddings, metadatas=metas)
        except Exception as e:
            print(f"Batch processing failed: {e}")

    # --- NEW: Re-ranking Optimized Search ---
    def search(
        self, 
        query: str, 
        k: int = 5,       # Final number of results you want
        fetch_k: int = 50 # Number of candidates to fetch from Vector DB (higher is better for re-ranking)
    ) -> List[Tuple[Any, float, Any]]:
        """
        1. Retrieval: Get top-N results from Vector DB.
        2. Re-ranking: Use Cross-Encoder to score relevance of (Query, Document).
        3. Return top-k sorted by re-ranker score.
        """
        # 1. Embed Query
        q_emb = self.embedder.embed(query)
        if not isinstance(q_emb, np.ndarray):
            q_emb = np.array(q_emb)
            
        # 2. Vector Retrieval (Fetch more candidates than needed)
        # We fetch 'fetch_k' (e.g., 50) candidates
        print(f"Retrieving top {fetch_k} candidates via Vector Search...")
        candidates = self.db.search(q_emb, k=fetch_k)
        
        if not candidates or self.reranker is None:
            return candidates[:k]

        # 3. Prepare for Re-ranking
        # CrossEncoder expects a list of pairs: [[query, doc1], [query, doc2], ...]
        rerank_inputs = []
        valid_candidates = [] # Keep track of candidates that actually have text
        
        for uid, score, meta in candidates:
            if meta and 'text' in meta:
                rerank_inputs.append([query, meta['text']])
                valid_candidates.append((uid, score, meta))
        
        if not rerank_inputs:
            return candidates[:k]

        # 4. Perform Re-ranking
        print(f"Re-ranking {len(rerank_inputs)} candidates...")
        # predict returns a numpy array of scores
        rerank_scores = self.reranker.predict(rerank_inputs)
        
        # 5. Combine and Sort
        # We create a new list of (uid, NEW_SCORE, meta)
        ranked_results = []
        for i, (uid, old_score, meta) in enumerate(valid_candidates):
            new_score = float(rerank_scores[i])
            ranked_results.append((uid, new_score, meta))
            
        # Sort by new score descending
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # 6. Return top k
        final_results = ranked_results[:k]
        
        return final_results

if __name__ == "__main__":
    # Test run
    indexer = Indexer()
    
    # 1. Run Indexing (Optional if already done)
    # indexer.index_all()
    
    # 2. Test Optimized Search
    query = "What is the financial outlook?"
    results = indexer.search(query, k=3, fetch_k=20)
    
    print(f"\n--- Top Results for '{query}' ---")
    for uid, score, meta in results:
        # Score is now the Cross-Encoder score (logits), usually between -10 and 10
        # Higher is better. >0 is generally relevant.
        print(f"\n[Re-rank Score: {score:.4f}] ID: {uid}")
        if meta and 'text' in meta:
            snippet = meta['text'][:150].replace('\n', ' ')
            print(f"Snippet: {snippet}...")
            print(f"Source: {meta.get('source', 'Unknown')}")