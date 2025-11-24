import os
import sqlite3
import faiss
import numpy as np
import pickle
import threading
from typing import Iterable, List, Any, Optional, Tuple, Dict, Union
from tqdm import tqdm

DEFAULT_INDEX_FILE = "hnsw.index"
DEFAULT_META_DB = "meta.sqlite3"
M_DEFAULT = 32
EF_CONSTRUCTION_DEFAULT = 200
EF_SEARCH_DEFAULT = 50

# Prevent tqdm thread warnings
tqdm.monitor_interval = 0

class FaissHNSWDB:
    def __init__(
        self,
        dim: int,
        path: str = "hnsw_store",
        M: int = M_DEFAULT,
        ef_construction: int = EF_CONSTRUCTION_DEFAULT,
        ef_search: int = EF_SEARCH_DEFAULT,
        metric: str = "cosine"
    ):
        """
        dim: vector dimension
        path: directory to persist index + sqlite
        M: HNSW connectivity parameter (typ. 16..64)
        ef_construction: HNSW build parameter (higher => higher recall, slower adds)
        ef_search: default ef for queries (higher => better recall, slower queries)
        metric: "cosine" or "l2"
        """
        self.dim = dim
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.index_file = os.path.join(self.path, DEFAULT_INDEX_FILE)
        self.meta_db = os.path.join(self.path, DEFAULT_META_DB)
        
        # Thread lock for write operations (add/delete)
        self.lock = threading.Lock()

        # HNSW parameters
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        if metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        self.metric = metric

        # sqlite connection
        self.conn = sqlite3.connect(self.meta_db, check_same_thread=False)
        self._init_tables()

        # create or load index
        if os.path.exists(self.index_file):
            print(f"Loading FAISS index from {self.index_file}")
            self.index = faiss.read_index(self.index_file)
            
            # Sanity checks
            if not isinstance(self.index, faiss.IndexIDMap2):
                # If it wasn't saved as IDMap, wrap it now
                print("Warning: Loaded index was not IDMap2, wrapping it.")
                self.index = faiss.IndexIDMap2(self.index)
            
            self._set_hnsw_parameters(self.ef_construction, self.ef_search)
        else:
            print(f"Creating new HNSW index (dim={self.dim}, M={self.M}, metric={self.metric})")
            self.index = self._make_index(self.dim, self.M, self.metric)
            
            # Ensure wrapper is IndexIDMap2 for custom IDs
            if not isinstance(self.index, faiss.IndexIDMap2):
                self.index = faiss.IndexIDMap2(self.index)
                
            self._set_hnsw_parameters(self.ef_construction, self.ef_search)

        # Initialize ID counter
        self._next_int_id = self._compute_next_int_id()

    def _init_tables(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS id_map (
                    user_id TEXT PRIMARY KEY,
                    int_id INTEGER UNIQUE,
                    metadata BLOB
                )
            """)
            self.conn.commit()

    def _make_index(self, dim: int, M: int, metric: str):
        if metric == "cosine":
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            metric_type = faiss.METRIC_L2

        # Create HNSW index
        return faiss.IndexHNSWFlat(dim, M, metric_type)

    def _compute_next_int_id(self) -> int:
        cur = self.conn.cursor()
        res = cur.execute("SELECT MAX(int_id) FROM id_map").fetchone()
        if res and res[0] is not None:
            return int(res[0]) + 1
        return 1

    def _serialize_meta(self, meta):
        return pickle.dumps(meta)

    def _set_hnsw_parameters(self, ef_construction: Optional[int] = None, ef_search: Optional[int] = None):
        """
        recursively digs into the index to find the HNSW structure and set parameters.
        This is more robust than checking .index.index.hnsw
        """
        # Helper to traverse down to the actual HNSW index
        index_node = self.index
        while hasattr(index_node, "index"):
            index_node = index_node.index
        
        # Now index_node should be the HNSWFlat (or similar)
        if hasattr(index_node, "hnsw"):
            if ef_construction:
                index_node.hnsw.efConstruction = int(ef_construction)
            if ef_search:
                index_node.hnsw.efSearch = int(ef_search)
        else:
            # Fallback: ParameterSpace (generic FAISS setter)
            ps = faiss.ParameterSpace()
            try:
                if ef_construction:
                    ps.set_index_parameter(self.index, "efConstruction", int(ef_construction))
                if ef_search:
                    ps.set_index_parameter(self.index, "efSearch", int(ef_search))
            except Exception as e:
                # Often fails on IDMaps, hence the manual traversal above is preferred
                pass

    def add(
        self,
        user_ids: Iterable[Any],
        vectors: np.ndarray,
        metadatas: Optional[Iterable[Any]] = None,
        normalize: bool = True,
        batch_size: int = 10000,
    ):
        """
        Add vectors with user-defined ids and metadata.
        Thread-safe via self.lock.
        """
        with self.lock:
            uids = list(user_ids)
            n = len(uids)

            if vectors.shape != (n, self.dim):
                raise ValueError(f"vectors must be (n, {self.dim}) but got {vectors.shape}")

            if metadatas is None:
                metadatas = [None] * n
            else:
                metadatas = list(metadatas)
                if len(metadatas) != n:
                    raise ValueError("Metadata count does not match vector count")

            # Convert to float32
            X = vectors.astype(np.float32, copy=False)

            # Normalize if cosine metric
            if self.metric == "cosine" and normalize:
                faiss.normalize_L2(X)

            cur = self.conn.cursor()

            clean_vectors = []
            clean_int_ids = []
            clean_metas_db = [] # Tuples for SQLite

            # 1. Filter duplicates & Prepare Data
            for uid, vec, meta in zip(uids, X, metadatas):
                uid_s = str(uid)
                
                # Check for duplicate ID in DB
                existing = cur.execute(
                    "SELECT int_id FROM id_map WHERE user_id = ?", (uid_s,)
                ).fetchone()

                if existing:
                    # Overwrite logic or Skip? 
                    # Current logic: Skip duplicates.
                    continue
                
                # Check for Zero Vector (prevents NaNs in index)
                if np.allclose(vec, 0):
                    print(f"Warning: Vector for {uid_s} is all zeros. Skipping.")
                    continue

                iid = self._next_int_id
                self._next_int_id += 1

                clean_vectors.append(vec)
                clean_int_ids.append(iid)
                # Store data for DB insertion later
                clean_metas_db.append((uid_s, iid, self._serialize_meta(meta)))

            if not clean_vectors:
                return

            X_clean = np.vstack(clean_vectors).astype(np.float32)
            ids_np = np.array(clean_int_ids, dtype=np.int64)

            # 2. Add to FAISS (Memory)
            # We do this BEFORE committing to DB. If this fails, DB is untouched.
            start = 0
            total = len(ids_np)
            try:
                while start < total:
                    end = min(total, start + batch_size)
                    xb = X_clean[start:end]
                    idb = ids_np[start:end]
                    self.index.add_with_ids(xb, idb)
                    start = end
            except Exception as e:
                print(f"Error adding to FAISS: {e}")
                # Rollback internal ID counter logic if needed, or just let it skip
                raise e

            # 3. Insert into SQLite (Disk/Meta)
            try:
                cur.executemany(
                    "INSERT INTO id_map (user_id, int_id, metadata) VALUES (?, ?, ?)",
                    clean_metas_db
                )
                self.conn.commit()
            except Exception as e:
                print(f"Error writing to SQLite: {e}")
                # Critical: Data mismatch. FAISS has it, DB doesn't.
                # Attempt to rollback FAISS (hard with HNSW) or reload.
                raise e

    def search(
        self, 
        query: np.ndarray, 
        k: int = 10, 
        ef_search: Optional[int] = None, 
        normalize: bool = True
    ) -> List[Tuple[Any, float, Any]]:
        faiss.omp_set_num_threads(1)
        
        # 1. CRITICAL: Force strict float32 and contiguous memory layout
        # FAISS segfaults if input is float64 or non-contiguous
        q = np.asarray(query, dtype=np.float32)
        if not q.flags['C_CONTIGUOUS']:
            q = np.ascontiguousarray(q, dtype=np.float32)
        
        # Handle 1D array (single vector)
        if q.ndim == 1:
            q = q.reshape(1, -1)
            
        if q.shape[1] != self.dim:
            raise ValueError(f"Query dimension mismatch. Expected {self.dim}, got {q.shape[1]}")

        # 2. Normalize safely
        if self.metric == "cosine" and normalize:
            faiss.normalize_L2(q)

        # 3. Safety Check: Is index empty?
        if self.index.ntotal == 0:
            print("Warning: Index is empty. Returning empty results.")
            return []

        # 4. Set Params
        if ef_search is not None:
            self._set_hnsw_parameters(ef_search=int(ef_search))

        # 5. Search
        actual_k = min(k, self.index.ntotal)
        D, I = self.index.search(q, actual_k)

        ids = I[0]
        scores = D[0]
        
        result = []
        cur = self.conn.cursor()
        
        for int_id, score in zip(ids, scores):
            if int_id == -1:
                continue
                
            row = cur.execute(
                "SELECT user_id, metadata FROM id_map WHERE int_id = ?", 
                (int(int_id),)
            ).fetchone()
            
            if row:
                uid, pm = row
                try:
                    meta = pickle.loads(pm) if pm is not None else None
                except Exception:
                    meta = {}
                result.append((uid, float(score), meta))
                
        return result

    def get(self, user_id: Any) -> Optional[Dict]:
        cur = self.conn.cursor()
        row = cur.execute("SELECT int_id, metadata FROM id_map WHERE user_id = ?", (str(user_id),)).fetchone()
        if not row:
            return None
        iid, pm = row
        meta = pickle.loads(pm) if pm is not None else None
        
        try:
            vec = self.index.reconstruct(int(iid))
        except Exception:
            # HNSW might fail reconstruction depending on exact build config, 
            # but HNSWFlat usually supports it.
            vec = None
            
        return {"user_id": str(user_id), "int_id": iid, "vector": vec, "metadata": meta}

    def delete(self, user_id: Any):
        """
        Removes a user from SQLite and attempts to remove from FAISS.
        """
        with self.lock:
            cur = self.conn.cursor()
            row = cur.execute("SELECT int_id FROM id_map WHERE user_id = ?", (str(user_id),)).fetchone()
            if not row:
                raise KeyError(f"{user_id} not found")
            
            iid = int(row[0])
            
            # Remove from DB
            cur.execute("DELETE FROM id_map WHERE user_id = ?", (str(user_id),))
            self.conn.commit()
            
            # Remove from FAISS
            try:
                self.index.remove_ids(np.array([iid], dtype=np.int64))
            except Exception as e:
                print(f"FAISS remove_ids failed (not supported by all index types): {e}")
                print("Triggering soft rebuild...")
                self._rebuild_index_from_sqlite()

    def _rebuild_index_from_sqlite(self):
        """
        Rebuilds the index from scratch using data stored in the existing index 
        (via reconstruction) to clean up gaps.
        """
        print("Rebuilding index...")
        cur = self.conn.cursor()
        rows = cur.execute("SELECT user_id, int_id FROM id_map ORDER BY int_id").fetchall()
        
        # Create fresh index
        fresh_idx = self._make_index(self.dim, self.M, self.metric)
        fresh_idx = faiss.IndexIDMap2(fresh_idx)
        
        if not rows:
            self.index = fresh_idx
            return

        # Extract vectors from OLD index
        vecs = []
        valid_ids = []
        
        for r in tqdm(rows, desc="Reconstructing"):
            iid = int(r[1])
            try:
                v = self.index.reconstruct(iid)
                vecs.append(v)
                valid_ids.append(iid)
            except Exception:
                print(f"Warning: Could not reconstruct vector for ID {iid}")

        if vecs:
            X = np.vstack(vecs).astype(np.float32)
            ids_np = np.array(valid_ids, dtype=np.int64)
            fresh_idx.add_with_ids(X, ids_np)

        # Swap
        self.index = fresh_idx
        self._set_hnsw_parameters(self.ef_construction, self.ef_search)
        print("Rebuild complete.")

    def save(self):
        with self.lock:
            print(f"Saving FAISS HNSW index to {self.index_file}")
            faiss.write_index(self.index, self.index_file)
            self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

# Demo test
if __name__ == "__main__":
    import shutil
    
    # Cleanup old demo
    if os.path.exists("demo_hnsw"):
        shutil.rmtree("demo_hnsw")

    dim = 128
    db = FaissHNSWDB(dim=dim, path="demo_hnsw", M=32, ef_construction=200, ef_search=50, metric="cosine")
    
    print("1. Creating data...")
    N = 1000
    X = np.random.randn(N, dim).astype(np.float32)
    ids = [f"user_{i}" for i in range(N)]
    metas = [{"info": f"meta_{i}"} for i in range(N)]
    
    print("2. Adding data...")
    db.add(ids, X, metas)
    
    print("3. Saving...")
    db.save()
    
    print("4. Searching...")
    q = np.random.randn(dim).astype(np.float32)
    results = db.search(q, k=5)
    
    print("Results found:", len(results))
    for r in results:
        print(r)
        
    print("5. Get Item...")
    item = db.get("user_0")
    print("Got item:", item['user_id'], item['metadata'])

    print("6. Deleting Item...")
    db.delete("user_0")
    
    print("Done.")