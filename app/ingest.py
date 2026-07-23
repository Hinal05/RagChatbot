"""Chunk the documents in data/ and store their embeddings in a local Chroma index.

Run this whenever data/ changes:
    python -m app.ingest
"""
import chromadb

from app.config import CHROMA_DIR, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.embeddings import embed_passages

COLLECTION_NAME = "drupal_docs"


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size])
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


def load_documents() -> list[tuple[str, str]]:
    """Returns list of (source_filename, full_text)."""
    docs = []
    for path in sorted(DATA_DIR.glob("*.md")):
        docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def build_index() -> int:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids, texts, metadatas = [], [], []
    for source, text in load_documents():
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{source}::{i}")
            texts.append(chunk)
            metadatas.append({"source": source, "chunk_index": i})

    if not texts:
        raise RuntimeError(f"No documents found in {DATA_DIR}. Add .md files first.")

    embeddings = embed_passages(texts)
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(texts)


if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} chunks into Chroma at {CHROMA_DIR}")
