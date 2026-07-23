import chromadb

from app.config import CHROMA_DIR, TOP_K, MAX_RELEVANT_DISTANCE
from app.embeddings import embed_query
from app.ingest import COLLECTION_NAME


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    hits = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        if distance > MAX_RELEVANT_DISTANCE:
            continue
        hits.append({"text": doc, "source": meta["source"], "distance": distance})
    return hits
