"""Embedding wrapper around a Sentence-Transformers model.

Model choice (intfloat/e5-base-v2) is documented in README.md under
"Step 1: Embedding Model Selection" — short version: strong MTEB retrieval
score, ~0.4GB footprint, runs comfortably on CPU, English-focused dataset.

e5 models require a "query: " / "passage: " prefix on the input text —
this is a documented quirk of the model family, not a bug.
"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_passages(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    prefixed = [f"passage: {t}" for t in texts]
    return model.encode(prefixed, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    model = get_embedding_model()
    return model.encode(f"query: {text}", normalize_embeddings=True).tolist()
