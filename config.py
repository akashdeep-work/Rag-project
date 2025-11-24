from pathlib import Path

# data + storage
DATA_DIR = Path("data")
STORE_DIR = Path("store")
STORE_DIR.mkdir(parents=True, exist_ok=True)

# FAISS / indexing
DIM = 768                    # default embedding dim (will be detected)
HNSW_M = 32
HNSW_EF_CONSTR = 200
HNSW_EF_SEARCH = 50
SHARD_SIZE = 2_000_000       # vectors per shard (adjust)

# batching and parallelism
EMBED_BATCH = 64
INDEX_ADD_BATCH = 2048
EXTRACTION_THREADS = 8
PROCESS_WORKERS = 4

# re-rank / retrieval
ANN_CANDIDATES = 200
RERANK_TOPK = 10

# LLM
USE_OPENAI = True            # if False, will return concatenated context + prompt instead of calling API
OPENAI_MODEL = "gpt-4o-mini" # change as desired

# FastAPI
HOST = "0.0.0.0"
PORT = 8000
