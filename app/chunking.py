"""Loads the knowledge base and splits it into embeddable chunks.

Strategy: entries in data/knowledge_base.json are short, atomic facts (1-4
sentences each), so most are embedded as a single chunk unchanged — splitting
a one-sentence entry would only lose context for no benefit. Only entries
longer than SHORT_ENTRY_WORD_THRESHOLD go through sliding-window splitting
(fixed-size chunks with overlap), which preserves continuity across a chunk
boundary at the cost of some duplicated text — an acceptable tradeoff for the
rare longer entry. See README.md "Step 2: Chunking Strategy" for the full
comparison against fixed-size-only and sentence-only splitting.
"""
import json

from app.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, SHORT_ENTRY_WORD_THRESHOLD

KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"


def load_knowledge_base() -> list[dict]:
    """Returns the list of {"id", "category", "text"} entries."""
    with open(KNOWLEDGE_BASE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _sliding_window(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
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


def chunk_entry(entry: dict) -> list[dict]:
    """Splits one knowledge-base entry into chunks tagged with its id/category."""
    text = entry["text"]
    pieces = [text] if len(text.split()) <= SHORT_ENTRY_WORD_THRESHOLD else _sliding_window(text)
    return [
        {
            "chunk_id": f"{entry['id']}::{i}",
            "text": piece,
            "category": entry["category"],
            "source_id": entry["id"],
        }
        for i, piece in enumerate(pieces)
    ]


def chunk_knowledge_base() -> list[dict]:
    """Loads and chunks every entry in the knowledge base."""
    chunks = []
    for entry in load_knowledge_base():
        chunks.extend(chunk_entry(entry))
    return chunks
