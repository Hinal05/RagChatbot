# Setup & Run Guide

This is a copy-paste runbook for getting the RAG chatbot running on a new machine.
For architecture/design details (why e5-base-v2, why Chroma, how guardrails work),
see [README.md](README.md).

## 1. Prerequisites

- **Python 3.10+** (developed on 3.12)
- **Ollama** — runs the local LLM, no API key/account needed
- No GPU required; everything runs on CPU

## 2. Get the code onto the new machine

Copy the whole project folder **except** `venv/`, `__pycache__/`, and `chroma_db/`
(these are machine-specific / regenerated — see "What not to copy" below).

```bash
git clone <your-repo-url> rag-chatbot
cd rag-chatbot
```

## 3. Install Ollama and pull models

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3            # ~2.2GB, the default model
ollama pull qwen2.5:0.5b    # ~400MB, a second small/fast model for the UI's model-switcher
```

Ollama must be running (`ollama serve`, or it starts automatically as a service
depending on install method) and reachable at `http://localhost:11434`.

## 4. Python environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env` if needed:

```
EMBEDDING_MODEL_NAME=intfloat/e5-base-v2
OLLAMA_MODEL=phi3              # default model; switch models at runtime in ui.py
OLLAMA_HOST=http://localhost:11434
```

## 6. Build the vector index

The knowledge base lives in `data/knowledge_base.json` (110 entries across 5
categories: html_css, javascript, react, nodejs, drupal). It must be embedded
into Chroma **once** (and again any time you change that file):

```bash
python -m app.ingest
```

This creates/updates the `chroma_db/` folder — a local, persistent vector store.
It is gitignored on purpose; regenerate it on every new machine instead of copying it.

## 7. Run the chatbot

**Terminal chat:**
```bash
python cli_chat.py
```

**Streamlit UI** (chat layout, model-switcher dropdown, streamed responses):
```bash
streamlit run ui.py
```
Opens automatically in your browser (usually http://localhost:8501).

**FastAPI web UI** (alternate interface, custom HTML page):
```bash
uvicorn main:app --reload --port 8001
```
Then open http://127.0.0.1:8001 in a browser.

## Quick command reference

| Task | Command |
|---|---|
| Activate venv | `source venv/bin/activate` |
| Install deps | `pip install -r requirements.txt` |
| Rebuild vector index | `python -m app.ingest` |
| Run in terminal | `python cli_chat.py` |
| Run Streamlit UI | `streamlit run ui.py` |
| Run FastAPI web app | `uvicorn main:app --reload --port 8001` |
| Pull a different Ollama model | `ollama pull <model-name>`, then either update `OLLAMA_MODEL` in `.env` or pick it from `ui.py`'s dropdown at runtime |

## What not to copy / commit

- `venv/` — recreate with `python3 -m venv venv` on the target machine.
- `chroma_db/` — regenerate with `python -m app.ingest`.
- `__pycache__/`, `*.pyc` — Python bytecode cache, safe to delete anytime.
- `.env` — machine-specific config; copy `.env.example` instead and fill it in.

## Troubleshooting

- **"Connection refused" to Ollama** → Ollama isn't running. Try `ollama serve`
  or check `systemctl status ollama`.
- **Empty/odd answers right after setup** → you likely skipped `python -m app.ingest`,
  so Chroma has no data indexed yet.
- **Slow first response** → the embedding model and Ollama model both need to
  load into memory on first use; subsequent requests are faster.
- **Model-switcher dropdown only shows one model** → you haven't pulled a second
  Ollama model yet — run `ollama pull qwen2.5:0.5b` (or any other model) and
  refresh the Streamlit page.
