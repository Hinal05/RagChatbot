# Drupal RAG Chatbot

A chatbot that answers Drupal developer questions using its own small knowledge
base, and can also check the live weather. It runs a multi-turn conversation
(it remembers earlier messages), and it double-checks its own answers before
showing them to you.

"RAG" stands for **Retrieval-Augmented Generation** — instead of only relying
on what the AI model already knows, the app first looks up relevant text from
its own documents, then asks the model to answer using that text. This makes
answers more accurate and lets you update the knowledge base without
retraining any model.

## How it all fits together

```
data/*.md               →  app/ingest.py   →  chroma_db/ (a local search database)
                                                    ↑
your question  →  app/chat.py  →  app/retriever.py (finds the most relevant text)
                        │
                        ├─ app/tools.py (checks the weather, only when needed)
                        │
                        └─ Ollama (the AI model) → a structured answer → app/guardrails.py (checks it)
```

In plain steps: your question comes in → the app decides what kind of message
it is → if needed, it searches its documents for relevant text → it asks the
local AI model to answer, using that text as reference → it double-checks the
answer's shape before showing it to you.

There are two ways to use it, both powered by the same underlying engine:
`cli_chat.py` (a simple terminal chat) and `main.py` (a small web page you can
open in a browser).

## Step 1: Picking the AI model that reads your documents

Before the chatbot can search documents, it needs to convert text into
numbers (called "embeddings") so it can compare how similar two pieces of
text are. This is a separate, smaller AI model from the one that writes the
actual answers.

**Model chosen: `intfloat/e5-base-v2`** (runs locally on your computer, no
internet or account needed).

| What we cared about | Why this model works |
|---|---|
| Accuracy | It scores well on MTEB (a public leaderboard that ranks these models), clearly better than older, simpler options like `all-MiniLM-L6-v2`. |
| Language | It's built for English, which matches our documents (Drupal developer docs written in English). If we ever needed multiple languages, `intfloat/multilingual-e5-base` is the equivalent model for that. |
| Speed & size | About 0.4GB — small enough to run comfortably on a normal computer, no graphics card required. |
| Other options considered | OpenAI's `text-embedding-ada-002` is easier to set up (just an API call) but costs money per use and needs an internet connection — we skipped it to keep this project free and fully offline. We also considered `all-MiniLM-L6-v2` (smaller and faster) but it's noticeably less accurate, and our chosen model is already fast enough. |

One quirk to know about: this model expects a small label added to text
before comparing it — `"query: "` in front of your question, and `"passage: "`
in front of stored document text (see `app/embeddings.py`). This isn't a bug —
it's just how this particular model was trained to work, and skipping it
makes search results noticeably worse.

## Step 2: The knowledge base (what the chatbot actually knows)

- **The documents**: `data/*.md` — four short files we wrote ourselves,
  covering Drupal coding standards, module development, security practices,
  and performance tips. We kept this small on purpose, so it's easy to check
  by hand whether an answer is actually correct.
- **Chunking**: long documents get cut into smaller pieces (about 500 words
  each, with some overlap between pieces so we don't accidentally cut a
  sentence in half) — see `app/ingest.py`. Each piece remembers which file it
  came from.
- **Where it's stored**: [Chroma](https://www.trychroma.com/), a simple
  local database for searching by meaning rather than exact words, saved on
  your computer in `chroma_db/`. We picked this over a cloud service like
  Pinecone because it needs no sign-up, no internet, and no cost — a good
  fit for a small personal project. If we ever wanted to move to a cloud
  database, only `app/retriever.py` would need to change.

## Step 3: How a conversation actually works

Here's what happens, step by step, every time you send a message
(all of this lives in `app/chat.py`'s `ChatSession` class):

1. **Basic safety check** (`app/guardrails.py`) — blocks empty messages,
   messages that are way too long, and obvious attempts to trick the AI
   (like "ignore your previous instructions") — before wasting any time
   asking the AI model about them.
2. **Figuring out what kind of message this is** — a quick, cheap call to
   the AI model sorts your message into one of four types: a **greeting**
   ("hi"), **small talk** ("thanks"), a **weather** question (with a city
   name), or a real **question**. This matters because:
   - Greetings and small talk get a short, friendly reply right away — no
     document searching involved.
   - Weather questions go straight to the weather tool, also skipping
     document search.
   - Only real questions actually search the documents.

   This step exists because of a real bug we found: earlier, every message
   (even just "Hi") triggered a document search, and the AI would awkwardly
   try to summarize random document snippets instead of just saying hello.
3. **Searching the documents** (only for real questions) — your question
   gets compared against all the stored document pieces, and the 4 most
   relevant ones are pulled out (`app/retriever.py`). If your message looks
   like a quick follow-up to the previous question (short, and using words
   like "it" or "that"), we add a bit of the previous question first, so
   searching still works — see "Things to be aware of" below for the limits
   of this trick.
4. **Getting the answer** — the local AI model (via
   [Ollama](https://ollama.com)) is given the relevant document text (if
   any), the weather result (if relevant), and your recent conversation
   history, then asked to reply in one fixed, structured format:
   an answer, whether it used a tool, and which document(s) it used. If no
   relevant document was found, the model is told to answer any Drupal
   question it can confidently answer from its own general knowledge (not
   just basic ones — this covers most Drupal topics beyond our small
   knowledge base), mention that the answer isn't from the curated docs,
   and only say "I don't know" when it's genuinely not confident — never
   make up a fake excuse (we once saw it falsely claim its training data
   was outdated, which simply wasn't true).

   We looked into having it search Drupal.org live for anything not in the
   local docs, but Drupal.org (and related subdomains like `api.drupal.org`)
   now sit behind an anti-bot JavaScript challenge — every plain request,
   even to `robots.txt`, returns a generic challenge page instead of real
   content. Rather than trying to defeat that protection, we rely on the
   AI model's own general Drupal knowledge for anything outside the local
   docs instead.
5. **Double-checking the answer's format** — the AI's reply gets checked to
   make sure it's actually valid, structured data. If it isn't, we ask the
   model again with a correction; if it still fails, we show a safe,
   generic message instead of something broken.
6. **Remembering the conversation** — each question and answer gets saved
   to memory for that session, and the last several exchanges are replayed
   into future questions, so things like "and what about performance?"
   after an earlier question still make sense.

## Setting it up

You'll need: Python 3.10 or newer, and [Ollama](https://ollama.com) installed
on your computer (this runs the AI model locally, for free).

```bash
# 1. Install Ollama and download a model (one-time, about 5GB)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b      # or a smaller, faster model like `ollama pull phi3`

# 2. Set up the Python environment
cd ~/projects/rag-chatbot
source venv/bin/activate      # if venv doesn't exist yet: python3 -m venv venv
pip install -r requirements.txt

# 3. Build the searchable document database
python -m app.ingest

# 4. Start chatting
python cli_chat.py
# — or open it in a browser instead —
uvicorn main:app --reload --port 8001   # then visit http://127.0.0.1:8001
```

If you're using a different (or smaller) Ollama model, set `OLLAMA_MODEL` in
the `.env` file to match (e.g. `OLLAMA_MODEL=phi3`).

## Things you can try asking

- "What's the correct hook naming convention in Drupal?" → answered from `coding_standards.md`.
- "How do I prevent SQL injection in a custom module?" → answered from `security_best_practices.md`.
- "What's the weather in Ahmedabad right now?" → uses the live weather tool, no documents needed.
- "What's the weather in Ahmedabad, and also how does Drupal handle CSRF protection?" → uses both the tool and document search in one go.
- Ask about security, then follow up with "and what about performance?" → tests whether it remembers the conversation.

## Things to be aware of (current limitations)

- **Figuring out message type isn't perfect.** It's one quick AI call, and
  with a small local model (like `phi3`) it can occasionally get confused —
  we actually saw the exact same question get classified correctly one time
  and incorrectly (as small talk) another time. A more robust setup would
  use the AI provider's built-in "function calling" feature instead of our
  simple approach, and/or a larger, more capable model.
- **Follow-up questions use a simple trick, not true understanding.** If
  your message is short (under 8 words) and comes after at least one earlier
  exchange, we just tack on your previous question before searching. This
  works for cases like "what about performance?" but won't catch longer or
  less obvious follow-ups. A more thorough solution would use a dedicated AI
  step to properly rewrite the question first.
- **Documents are split by word count, not by meaning.** This is fine for
  our short files, but a much bigger set of documents would benefit from
  smarter splitting (e.g. by sentence or topic).
- **Conversations aren't saved permanently.** Everything is remembered only
  while the app is running, and is lost if you restart it. The web version
  automatically forgets a conversation after 30 minutes of inactivity —
  there's no permanent storage or database for chat history.
</content>
