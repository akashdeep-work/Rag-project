import argparse
from indexer import Indexer


def main():
    parser = argparse.ArgumentParser(description="RAG Indexer and Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ------------------------------
    # index command
    # ------------------------------
    idx = subparsers.add_parser("index", help="Index all documents")
    idx.add_argument("--chunk_size", type=int, default=500, help="Chunk size")
    idx.add_argument("--chunk_overlap", type=int, default=50, help="Chunk overlap")

    # ------------------------------
    # search command
    # ------------------------------
    sch = subparsers.add_parser("search", help="Search indexed DB")
    sch.add_argument("query", type=str, help="Query text")
    sch.add_argument("--k", type=int, default=20, help="Number of results")

    args = parser.parse_args()

    # Initialize indexer
    indexer = Indexer()

    # Handle commands
    if args.command == "index":
        print("🔍 Starting indexing...")
        indexer.index_all(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        print("✅ Indexing completed.")

    elif args.command == "search":
        print(f"🔎 Searching for: {args.query}")
        results = indexer.search(args.query, k=args.k)
        for uid, score, meta in results:
            print("\n-----------------------------")
            print("ID:", uid)
            print("Score:", score)
            print("Source:", meta.get("source"))
            print("Chunk Index:", meta.get("chunk_index"))
            print("Text:", meta.get("text"))

    else:
        parser.print_help()


if __name__ == "__main__":
    # import multiprocessing
    # multiprocessing.set_start_method("fork", force=True)
    main()
