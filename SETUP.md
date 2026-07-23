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

## 3. Install Ollama and pull a model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b     # ~5GB, best quality
# or a lighter model if resources are limited:
ollama pull phi3
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
OLLAMA_MODEL=llama3.1:8b        # match whatever model you pulled in step 3
OLLAMA_HOST=http://localhost:11434
```

## 6. Build the vector index

The knowledge base lives in `data/*.md`. It must be embedded into Chroma
**once** (and again any time you change the files in `data/`):

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

**Web UI:**
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
| Run web app | `uvicorn main:app --reload --port 8001` |
| Pull a different Ollama model | `ollama pull <model-name>` then update `OLLAMA_MODEL` in `.env` |

## What not to copy / commit

- `venv/` — recreate with `python3 -m venv venv` on the target machine (it's
  currently tracked in git despite being in `.gitignore`; worth removing from
  git history with `git rm -r --cached venv` at some point).
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
</content>
</invoke>
