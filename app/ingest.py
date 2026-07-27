"""Embed the knowledge base and store it in a local Chroma index.

Run this whenever data/knowledge_base.json changes:
    python -m app.ingest
"""
import chromadb

from app.config import CHROMA_DIR
from app.chunking import chunk_knowledge_base
from app.embeddings import embed_passages

COLLECTION_NAME = "web_dev_kb"


def build_index() -> int:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    chunks = chunk_knowledge_base()
    if not chunks:
        raise RuntimeError("No entries found in data/knowledge_base.json.")

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [{"category": c["category"], "source_id": c["source_id"]} for c in chunks]

    embeddings = embed_passages(texts)
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(texts)


if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} chunks into Chroma at {CHROMA_DIR}")
