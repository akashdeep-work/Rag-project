import os
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Generator, Dict, Any
import numpy as np
from tqdm import tqdm

# Ensure these match your actual config/file structure
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
    def __init__(self, db_store_dir: str = "rag_store", embed_model: str = "all-MiniLM-L6-v2"):
        self.db_dir = db_store_dir
        
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
            metric="cosine"
        )

    def _list_sources(self) -> List[str]:
        """Scans DATA_DIR for valid files and urls.txt."""
        files = []
        if not os.path.exists(DATA_DIR):
            print(f"Warning: {DATA_DIR} does not exist.")
            return files
            
        p_dir = Path(DATA_DIR)
        
        # 1. Scan Files
        for p in p_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".pdf", ".docx", ".txt"]:
                files.append(str(p.resolve()))
        
        # 2. Scan URLs
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
        """
        Reads content from a file path or URL.
        Returns: (source_hash_id, extracted_text)
        """
        try:
            # Handle URLs
            if source.startswith("http://") or source.startswith("https://"):
                text = read_url(source)
                src_id = make_hash(source)
                return src_id, text
            
            # Handle Files
            p = Path(source)
            suffix = p.suffix.lower()
            
            if suffix == ".pdf":
                text = read_pdf(p)
            elif suffix == ".docx":
                text = read_docx(p)
            elif suffix == ".txt":
                text = read_txt(p)
            else:
                return None, None

            # Generate ID based on file path string
            src_id = make_hash(str(p.resolve()))
            return src_id, text
            
        except Exception as e:
            print(f"Failed to extract {source}: {e}")
            return None, None

    def _chunk_generator(self, sources: List[str], chunk_size: int, chunk_overlap: int) -> Generator[Dict[str, Any], None, None]:
        """
        Yields chunks one by one to avoid holding full text in memory.
        """
        with ThreadPoolExecutor(max_workers=EXTRACTION_THREADS) as ex:
            # Submit all extraction jobs
            future_to_src = {ex.submit(self._extract, s): s for s in sources}
            
            for future in as_completed(future_to_src):
                src = future_to_src[future]
                try:
                    sid, text = future.result()
                    if not text or not text.strip():
                        continue
                        
                    # Chunk the text immediately
                    chunks = chunk_text(text, chunk_size, chunk_overlap)
                    
                    for i, c in enumerate(chunks):
                        # Construct a unique ID for this chunk
                        cid = chunk_id(sid, i)
                        yield {
                            "id": cid,
                            "text": c,
                            "source_id": sid,
                            "source_name": src,
                            "chunk_index": i
                        }
                except Exception as e:
                    print(f"Error processing {src}: {e}")

    def index_all(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Main pipeline: List -> Extract -> Chunk -> Embed -> Save
        Processing is done in batches to manage memory.
        """
        sources = self._list_sources()
        print(f"Found {len(sources)} sources to process.")
        if not sources:
            return

        chunk_gen = self._chunk_generator(sources, chunk_size, chunk_overlap)
        
        # Batches for embedding/insertion
        current_batch_texts = []
        current_batch_ids = []
        current_batch_metas = []
        
        total_added = 0

        # We need a progress bar, but generators don't have length. 
        # We'll just count processed chunks.
        pbar = tqdm(desc="Indexing Chunks", unit="chunk")

        for item in chunk_gen:
            current_batch_ids.append(item['id'])
            current_batch_texts.append(item['text'])
            
            # Prepare metadata dict for DB
            meta = {
                "text": item['text'], # Storing text in metadata allows retrieval
                "source": item['source_name'],
                "chunk_index": item['chunk_index']
            }
            current_batch_metas.append(meta)
            
            # When batch is full, Embed -> Add -> Clear
            if len(current_batch_texts) >= INDEX_ADD_BATCH:
                self._process_batch(current_batch_ids, current_batch_texts, current_batch_metas)
                total_added += len(current_batch_texts)
                pbar.update(len(current_batch_texts))
                
                # Reset buffers
                current_batch_ids = []
                current_batch_texts = []
                current_batch_metas = []
                
                # Force garbage collection occasionally
                if total_added % (INDEX_ADD_BATCH * 5) == 0:
                    gc.collect()

        # Process remaining items
        if current_batch_texts:
            self._process_batch(current_batch_ids, current_batch_texts, current_batch_metas)
            total_added += len(current_batch_texts)
            pbar.update(len(current_batch_texts))

        pbar.close()
        print(f"Indexing complete. Total chunks added: {total_added}")
        
        # Persist to disk
        self.db.save()

    def _process_batch(self, ids: List[str], texts: List[str], metas: List[dict]):
        """
        Helper to embed texts and push to DB.
        """
        try:
            # 1. Generate Embeddings
            embeddings = self.embedder.embed_texts(texts, batch_size=EMBED_BATCH)
            
            # 2. Add to Vector DB
            # FaissHNSWDB.add handles duplication checks internally
            self.db.add(
                user_ids=ids, 
                vectors=embeddings, 
                metadatas=metas
            )
        except Exception as e:
            print(f"Batch processing failed: {e}")

    def search(self, query: str, k: int = 10) -> List[Tuple[Any, float, Any]]:
        """
        End-to-end search: Query -> Embed -> Search DB
        """
        # 1. Embed Query
        # Ensure input is a list for the embedder if required, though 'embed' usually handles str
        # If your embedder.embed returns a 1D array, reshape it.
        q_emb = self.embedder.embed(query)
        
        # Ensure it's a numpy array
        if not isinstance(q_emb, np.ndarray):
            q_emb = np.array(q_emb)
            
        # 2. Search
        print(f"Searching for: '{query}' (k={k})")
        results = self.db.search(q_emb, k=k)
        
        return results

if __name__ == "__main__":
    # Test run
    indexer = Indexer()
    
    # 1. Run Indexing
    indexer.index_all()
    
    # 2. Test Search
    results = indexer.search("What is the financial outlook?", k=3)
    for uid, score, meta in results:
        print(f"\n[Score: {score:.4f}] {uid}")
        if meta and 'text' in meta:
            print(f"Text snippet: {meta['text'][:100]}...")