import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Embedding model — see README.md "Step 1: Embedding Model Selection" for justification.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/e5-base-v2")

# Local LLM served by Ollama (https://ollama.com). Pull with: ollama pull llama3.1:8b
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4
MAX_HISTORY_TURNS = 6

# Cosine distance (0 = identical, 2 = opposite) above which a retrieved chunk is
# considered irrelevant and dropped, so greetings/small talk don't drag in
# unrelated document content. Tuned for e5-base-v2 embeddings.
MAX_RELEVANT_DISTANCE = 0.27
