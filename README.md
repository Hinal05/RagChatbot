# Drupal RAG Chatbot

An end-to-end retrieval-augmented generation (RAG) chat system: a custom Drupal
knowledge base, local embeddings, a local LLM, multi-turn conversation,
guardrailed structured output, and one live external action (weather lookup).

## Architecture

```
data/*.md               →  app/ingest.py   →  chroma_db/ (local vector store)
                                                    ↑
user question  →  app/chat.py  →  app/retriever.py (retrieve top-k chunks)
                        │
                        ├─ app/tools.py (weather, only if the model routes to it)
                        │
                        └─ Ollama (local LLM) → structured JSON answer → app/guardrails.py
```

Two front ends sit on top of the same `ChatSession` engine: `cli_chat.py` (terminal)
and `main.py` (FastAPI + a minimal HTML page).

## Step 1: Embedding Model Selection

**Chosen model: `intfloat/e5-base-v2`** (via `sentence-transformers`, run locally).

| Factor | Notes |
|---|---|
| Performance | e5-base-v2 scores strongly on MTEB's retrieval average — noticeably ahead of older baselines like `all-MiniLM-L6-v2` while staying small enough for CPU inference. |
| Language coverage | English-only, which matches this dataset (Drupal developer docs). If the knowledge base were multilingual, `intfloat/multilingual-e5-base` would be the equivalent pick. |
| Latency / size | ~0.4GB, runs in a few hundred ms per batch on CPU — no GPU required, fits the "local, free, one-week individual project" constraint. |
| Alternative considered | OpenAI `text-embedding-ada-002` — better zero-setup convenience via API, but rejected here to keep the whole pipeline free, offline-capable, and inspectable end-to-end (no external cost/latency dependency for a learning exercise). |
| Alternative considered | `all-MiniLM-L6-v2` — smaller and faster, but lower retrieval accuracy; not worth the tradeoff given e5-base-v2 still runs comfortably on CPU. |

Quirk worth documenting: e5 models expect a `"query: "` prefix on search queries
and a `"passage: "` prefix on indexed documents — see `app/embeddings.py`. Without
the prefixes, retrieval quality drops noticeably; this is a documented property
of the e5 family, not implementation error.

## Step 2: Knowledge Base & Vector Store

- **Dataset**: `data/*.md` — four short Drupal reference docs (coding standards,
  module development, security best practices, performance tuning). Small and
  hand-authored so retrieved answers are easy to verify by eye.
- **Chunking**: word-based sliding window, 500 words per chunk, 80-word overlap
  (`app/ingest.py`), each chunk tagged with its source filename.
- **Vector store**: [Chroma](https://www.trychroma.com/), persisted locally to
  `chroma_db/`. Chosen over Pinecone for this project because it needs no account,
  no network calls, and no ongoing cost — appropriate for an individual, one-week
  assignment; the retrieval code (`app/retriever.py`) is the only place a swap to
  a cloud vector DB would require changes.

## Step 3: Chat Engine — Multi-turn, Tool Use, Guardrails

`app/chat.py`'s `ChatSession`:

1. **Input guardrail** (`app/guardrails.py`) — rejects empty/oversized messages
   and simple prompt-injection attempts (e.g. "ignore previous instructions")
   before spending a model call on them.
2. **Intent routing** — a first, cheap model call classifies the message into
   `greeting`, `chitchat`, `weather` (with a location), or `question`
   (`app/chat.py`, `_classify_intent`). This decides what work the rest of the
   turn actually needs to do:
   - `greeting`/`chitchat` skip Chroma retrieval entirely and get a short,
     direct reply — no document context is ever built for these.
   - `weather` calls `app/tools.py` (the free Open-Meteo API, no key required)
     and skips retrieval too.
   - `question` is the only path that runs retrieval.

   This replaces an earlier design that ran retrieval unconditionally for
   every message and only avoided the tool call by routing. That older
   design produced a real bug: asking "Hi" retrieved four unrelated document
   chunks and the model dutifully summarized them instead of greeting back.
3. **Retrieval** (`question` intent only) — the user's question (widened with
   the prior turn's question for short, pronoun-heavy follow-ups — see
   "Known limitations") is embedded and the top-4 most similar chunks above a
   relevance cutoff are pulled from Chroma (`app/retriever.py`).
4. **Answer generation** — the LLM (local, via [Ollama](https://ollama.com))
   is given the retrieved context, any tool result, and recent conversation
   history, and is instructed to reply with **only** a JSON object matching a
   fixed schema (`answer`, `used_tool`, `sources`). When retrieval finds
   nothing, the prompt explicitly tells the model to either answer basic
   general-knowledge questions directly (e.g. "What is Drupal?") or say
   plainly that the knowledge base doesn't cover it — and never to invent a
   reason (like a fabricated training-cutoff date) for not knowing something.
5. **Output guardrail** — the JSON is parsed and validated against a Pydantic
   schema; if it fails to parse, the engine retries once with an explicit
   correction before falling back to a safe default message. This prevents
   malformed or off-schema output from ever reaching the user.
6. **Multi-turn memory** — each turn's user message and answer are appended to
   `self.history` and the last `MAX_HISTORY_TURNS` turns are replayed into the
   next call, so follow-up questions ("what about the second one?") work.

## Setup

Requires: Python 3.10+, and [Ollama](https://ollama.com) installed locally.

```bash
# 1. Install Ollama and pull a model (one-time, ~5GB download)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b      # or a smaller model, e.g. `ollama pull phi3`

# 2. Python environment
cd ~/projects/rag-chatbot
source venv/bin/activate      # venv already exists; recreate with `python3 -m venv venv` if needed
pip install -r requirements.txt

# 3. Build the vector index from data/
python -m app.ingest

# 4. Chat
python cli_chat.py
# — or, for the web UI —
uvicorn main:app --reload --port 8001   # then open http://127.0.0.1:8001
```

If you use a smaller/different Ollama model, set `OLLAMA_MODEL` in `.env`
(e.g. `OLLAMA_MODEL=phi3`).

## Example interactions to try

- "What's the correct hook naming convention in Drupal?" → RAG answer from `coding_standards.md`.
- "How do I prevent SQL injection in a custom module?" → RAG answer from `security_best_practices.md`.
- "What's the weather in Ahmedabad right now?" → triggers the external tool, no document lookup needed.
- "What's the weather in Ahmedabad, and also how does Drupal handle CSRF protection?" → exercises both retrieval and the tool in one turn.
- Follow-up: "and what about performance?" (after asking about security) → tests multi-turn memory.

## Known limitations

- Intent routing is a single extra LLM call classifying `greeting` /
  `chitchat` / `weather` / `question` — with a small local model (e.g.
  `phi3`) it can occasionally misclassify a real question as chitchat or vice
  versa; this was observed directly during testing (the same question
  classified correctly on one run and as chitchat on another). A production
  system would use the model provider's native function-calling API instead
  of this hand-rolled classification prompt, and/or a larger, more reliable
  model.
- Follow-up retrieval uses a simple heuristic — short (<8 words) messages
  after at least one turn of history get the prior user question prepended
  before embedding, so "what about performance?" retrieves performance
  content. This won't catch longer or less pronoun-obvious follow-ups; a
  more robust design would use a dedicated query-rewriting model call.
- The chunker is word-count based, not sentence/semantic aware — fine for this
  dataset's short docs, but a larger corpus would benefit from a smarter splitter.
- Conversation history is in-memory only (per `ChatSession` instance) and is
  lost on process restart; the FastAPI app keys sessions by a UUID with a
  simple 30-minute idle eviction policy (`main.py`) rather than a real
  persistence/session store.
