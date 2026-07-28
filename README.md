# Web Development RAG Chatbot

A chatbot that answers web development questions — HTML/CSS, JavaScript,
React, Node.js, and Drupal — using its own curated knowledge base, and can
also check the live weather. It runs a multi-turn conversation (it remembers
earlier messages), and it double-checks its own answers before showing them
to you.

"RAG" stands for **Retrieval-Augmented Generation** — instead of only relying
on what the AI model already knows, the app first looks up relevant text from
its own documents, then asks the model to answer using that text. This makes
answers more accurate and lets you update the knowledge base without
retraining any model.

## How it all fits together

```
data/knowledge_base.json →  app/chunking.py  →  app/ingest.py  →  chroma_db/ (a local search database)
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

There are three ways to use it, all powered by the same underlying engine
(`app/chat.py`'s `ChatSession`): `cli_chat.py` (a simple terminal chat),
`main.py` (a small hand-written FastAPI + HTML web page), and `ui.py` (a
Streamlit chat app with a model-switcher and streamed responses — see
Step 4 below).

## Step 1: Picking the AI model that reads your documents

Before the chatbot can search documents, it needs to convert text into
numbers (called "embeddings") so it can compare how similar two pieces of
text are. This is a separate, smaller AI model from the one that writes the
actual answers.

**Model chosen: `intfloat/e5-base-v2`** (runs locally on your computer, no
internet or account needed).

We picked this from the [MTEB leaderboard](https://huggingface.co/blog/mteb),
Hugging Face's public benchmark that scores embedding models across 56
datasets and up to 112 languages, so models can be compared on the same
scale rather than by marketing claims.

| What we cared about | Why this model works |
|---|---|
| Accuracy | Scores well on MTEB's retrieval average, clearly ahead of older, simpler options like `all-MiniLM-L6-v2`. We treated the MTEB score as a starting filter, not the final word — [Pinecone's embedding model guide](https://www.pinecone.io/learn/series/rag/embedding-models-rundown/) points out that some open-source models are effectively fine-tuned on the MTEB test sets themselves, which can inflate their leaderboard numbers, so real usage still matters more than the number alone. |
| Language | Built for English, which matches our knowledge base (web development reference content written in English). If we ever needed multiple languages, `intfloat/multilingual-e5-base` is the equivalent model for that. |
| Sequence length | Supports up to 512 tokens per chunk, which the same Pinecone guide notes is "usually more than enough" for typical retrieval chunks — comfortably covers our ~500-word chunk size (see Step 2). |
| Speed & size | About 0.4GB — small enough to run comfortably on a normal computer, no graphics card required. Pinecone's own side-by-side timing test found E5 embedded its test set in about 3 minutes 53 seconds, faster than Cohere's model (5:32) and OpenAI's `text-embedding-ada-002` (9:07) in their comparison — and confirms E5 doesn't strictly need a GPU, CPU-only is fine, just slower. |
| Other options considered | OpenAI's `text-embedding-ada-002` is easier to set up (just an API call) but costs money per use, needs an internet connection, and produces larger 1536-dimension vectors (more storage) — we skipped it to keep this project free, offline, and lightweight to store. We also considered `all-MiniLM-L6-v2` (smaller and faster) but it's noticeably less accurate, and E5 is already fast enough on CPU that the speed difference isn't worth the accuracy tradeoff. |

One quirk to know about: this model expects a small label added to text
before comparing it — `"query: "` in front of your question, and `"passage: "`
in front of stored document text (see `app/embeddings.py`). This isn't a bug —
it's just how this particular model was trained to work, and skipping it
makes search results noticeably worse.

## Step 2: Document Chunking Strategy

Splitting text into pieces ("chunks") is necessary because embedding
models and LLMs have a limited number of tokens they can read at once —
[NVIDIA's RAG 101 guide](https://developer.nvidia.com/blog/rag-101-demystifying-retrieval-augmented-generation-pipelines/)
notes that a model like `e5-large-v2` maxes out around 512 tokens, and
calls text-splitting "a nuanced process" rather than a trivial one. We
looked at the common approaches before picking ours:

| Approach | What we found |
|---|---|
| Fixed-size / character count | Simplest to implement, but cuts text at arbitrary points with no regard for sentence structure. A [chunking strategies overview](https://medium.com/@rahulpant.me/chunking-text-splitting-strategies-llms-579ab4ede2eb) calls this "a rarely used approach" in practice, more of a fallback than a real strategy. |
| Sentence/paragraph splitting | Respects natural language boundaries, but chunks can end up very uneven in size — some too short to be useful, some still too long. |
| Sliding window (overlapping fixed-size chunks) | [Analytics Vidhya's chunking guide](https://www.analyticsvidhya.com/blog/2024/10/chunking-techniques-to-build-exceptional-rag-systems/) notes this "preserves context across chunks" and "reduces information loss at chunk boundaries," at the cost of some redundant, repeated text and extra chunks to store. |
| Semantic splitting (ML-based) | Produces the most contextually meaningful chunks by splitting where topics actually shift, but is "computationally expensive" and adds real implementation complexity for not much benefit on our short entries. |

**What we actually do (`app/chunking.py`)**: since our knowledge base is
already made of short, atomic facts rather than long documents, most
entries (anything ≤120 words) are embedded as a **single chunk, unchanged**
— splitting a one-sentence fact would only lose context for no benefit.
Any entry longer than that threshold falls back to **sliding-window**
chunking (500 words, 80-word overlap) — chosen over fixed-size-only because
of the context-preservation benefit above, and considered a safe, simple
default given how rarely it's actually triggered by this dataset. This
hybrid is a deliberate, documented tradeoff: simplicity and speed for the
common case (short entries), with the more careful sliding-window approach
reserved for the rare long one.

## Step 3: Dataset Creation & Curation

**Domain: Web Development.** `data/knowledge_base.json` holds **110 short,
hand-written facts across 5 sub-categories** — `html_css`, `javascript`,
`react`, `nodejs`, and `drupal` (22 entries each). Each entry is a
self-contained fact or best practice, 1-4 sentences long, tagged with a
`category` field, e.g.:

```json
{"id": "react_005", "category": "react", "text": "The useEffect hook runs side effects..."}
```

We chose these 5 categories because they're genuinely different
technologies, not just different tones of writing about the same thing —
HTML/CSS (markup and styling), JavaScript (core language), React
(component-based UI), Node.js (server-side JS), and Drupal (a full CMS
built on PHP) each have their own concerns, conventions, and vocabulary,
which is a more meaningful kind of diversity than just varying the writing
style of one topic. Weather is deliberately **not** a category here — it's
handled entirely by the live tool (see the chat engine section below), so
retrieval (searching stored facts) and the live action (calling an API)
stay clearly separate from each other.

**Where it's stored**: [Chroma](https://www.trychroma.com/), a simple local
database for searching by meaning rather than exact words, saved on your
computer in `chroma_db/`. Every chunk carries its `category` and
originating entry `id` as metadata, so retrieval results can always be
traced back to a specific category and fact. We picked Chroma over a cloud
service like Pinecone because it needs no sign-up, no internet, and no
cost — a good fit for a small personal project. If we ever wanted to move
to a cloud database, only `app/retriever.py` would need to change.

## Step 4: The Streamlit UI (chat layout, model switching, streaming)

`ui.py` is a small [Streamlit](https://streamlit.io) app built on top of the
exact same `ChatSession` engine as `cli_chat.py` and `main.py` — it doesn't
duplicate any logic, it just wraps it in a chat UI. Run it with
`streamlit run ui.py`.

- **Chat layout**: uses Streamlit's built-in `st.chat_message`/`st.chat_input`
  components, which are purpose-built for exactly this — a scrolling
  conversation with a message box at the bottom.
- **Model switching**: a sidebar dropdown lets you switch between installed
  Ollama models mid-conversation. The dropdown is populated dynamically
  from `ollama.Client().list()` (not hardcoded), so it reflects whatever's
  actually installed. Out of the box this project pulls two: `phi3`
  (~2.2GB, more reliable) and `qwen2.5:0.5b` (~400MB, much faster, but
  noticeably less reliable — while testing, the tiny model sometimes
  misclassified a plain factual question as small talk, something `phi3`
  rarely does; that's a genuine, honest illustration of why model choice
  matters, not a bug we hid). Switching models keeps the conversation
  history intact — only which model answers the *next* message changes.
  - **Why local models, not OpenRouter/LiteLLM**: the assignment suggests
    routing across hosted providers (GPT-4, Claude, Gemini) via OpenRouter
    or LiteLLM. We looked at both (see the linked OpenRouter-Streamlit
    example and LiteLLM's `"openrouter/<provider>/<model>"` routing
    pattern) but kept this project's existing free/local/no-API-key design
    instead — switching between local Ollama models is the same underlying
    idea (pick an identifier, route the request to it), just without
    needing paid API keys.
- **Streaming**: real token-by-token streaming for the AI-generated
  question path, via Ollama's `stream=True` option (see
  `ChatSession.ask_stream` in `app/chat.py`). One tradeoff worth being
  upfront about: since the model is instructed to output one structured
  JSON object (Step 6's guardrail), the UI briefly shows that raw JSON
  forming character-by-character before swapping in the final, cleanly
  parsed answer. We considered streaming a separate plain-text-only call
  instead to avoid this, but rejected it — that would mean the text you
  see streaming isn't the same data actually being schema-validated,
  which defeats the point of the structured-output guardrail. Greetings,
  small talk, and weather answers are already instant (no LLM call, or a
  single quick one), so they just appear immediately rather than streaming.
- **Instructions**: the sidebar explains what the chatbot can answer (the 5
  categories, plus the live weather lookup) and that weather is detected
  automatically from your message — there's no special command to type.

## Step 5: The Chat Engine (Multi-Turn Conversations & Multi-Stage Prompting)

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
   relevant document was found, the model is told to answer any web
   development question it can confidently answer from its own general
   knowledge (not just basic ones — this covers most topics beyond our
   small knowledge base), mention that the answer isn't from the curated
   docs, and only say "I don't know" when it's genuinely not confident —
   never make up a fake excuse (we once saw it falsely claim its training
   data was outdated, which simply wasn't true).

   We looked into having it search Drupal.org live for anything not in the
   local docs, but Drupal.org (and related subdomains like `api.drupal.org`)
   now sit behind an anti-bot JavaScript challenge — every plain request,
   even to `robots.txt`, returns a generic challenge page instead of real
   content. Rather than trying to defeat that protection, we rely on the
   AI model's own general knowledge for anything outside the local docs
   instead.
5. **Double-checking the answer's format** — the AI's reply gets checked to
   make sure it's actually valid, structured data. If it isn't, we ask the
   model again with a correction; if it still fails, we show a safe,
   generic message instead of something broken.
6. **Remembering the conversation** — each question and answer gets saved
   to memory for that session, and the last several exchanges are replayed
   into future questions, so things like "and what about performance?"
   after an earlier question still make sense.

## Step 6: Prompt Engineering & Guardrails

- **System prompts** (`app/chat.py`): four separate, narrowly-scoped system
  prompts — one for intent classification, one for weather-location
  extraction, one for greetings/small talk, and the main
  `ANSWER_SYSTEM_PROMPT` that governs real answers. Each defines the AI's
  role, exactly what shape to respond in, and domain-specific rules (e.g.
  when to fall back to general knowledge vs. say "I don't know").
- **Few-shot examples** (`ANSWER_FEW_SHOT_EXAMPLES` in `app/chat.py`): 4
  demonstration user/assistant turns inserted before every real question,
  covering the exact behaviors the assignment asks for:
  1. A correctly grounded answer that cites its source.
  2. Honestly saying "I don't know" for something too specific to be
     confident about, instead of guessing.
  3. A neutral, non-engaging reply to a rude/insulting message.
  4. Refusing a request to help with something harmful (e.g. writing an
     attack script).

  **Verified difference this made**: before adding these examples, asking
  "You are the worst AI ever, you are useless." produced *"Sorry, I
  couldn't produce a valid structured answer"* — the model tried to
  apologize and engage conversationally in plain text, which didn't match
  the required JSON shape at all. After adding the few-shot examples, the
  same message produces a clean, in-schema, neutral deflection like *"My
  goal is to assist with web development questions and I'll do my best
  within that scope."* This is the same before/after pattern the
  assignment's own guardrails article describes.
- **Structured output**: every real answer must be a JSON object validated
  against a Pydantic model (`ChatAnswer` in `app/guardrails.py` — `answer`,
  `used_tool`, `sources`), not free text. This is what makes it possible to
  reliably display "used external tool" / "sources: ..." in both UIs
  without guessing at the text.
- **Guardrails**:
  - *Input guardrail* (`input_guardrail`): blocks empty messages, messages
    over 4000 characters, and simple prompt-injection attempts ("ignore
    previous instructions") before any model call is made.
  - *Prompt-level guardrails*: the rude-input and harmful-request rules
    above, reinforced by the few-shot examples — this is where the model
    itself is steered, rather than a hardcoded keyword block, since tone
    (not a fixed disallowed-words list) is what actually needs handling
    here.
  - *Output guardrail*: covered below under parsing.
- **Parsing logic** (`parse_structured_answer` in `app/guardrails.py`):
  strips markdown code fences if present, extracts the `{...}` span, then
  parses with `json.loads(..., strict=False)` — the `strict=False` flag
  was added after finding a real bug: small local models sometimes emit a
  literal newline inside a JSON string value (instead of an escaped `\n`),
  which strict JSON parsing rejects outright. If parsing or Pydantic
  validation still fails, `app/chat.py` retries once with an explicit
  correction message; if that also fails, it falls back to a safe generic
  message rather than ever showing something broken to the user.

## Setting it up

You'll need: Python 3.10 or newer, and [Ollama](https://ollama.com) installed
on your computer (this runs the AI model locally, for free).

```bash
# 1. Install Ollama and download two models (one-time)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3             # ~2.2GB, the default
ollama pull qwen2.5:0.5b     # ~400MB, a second model so ui.py's switcher has options

# 2. Set up the Python environment
cd ~/projects/rag-chatbot
source venv/bin/activate      # if venv doesn't exist yet: python3 -m venv venv
pip install -r requirements.txt

# 3. Build the searchable document database
python -m app.ingest

# 4. Start chatting
python cli_chat.py
# — or the Streamlit UI, with model switching and streaming —
streamlit run ui.py
# — or the FastAPI web page —
uvicorn main:app --reload --port 8001   # then visit http://127.0.0.1:8001
```

If you're using a different Ollama model, set `OLLAMA_MODEL` in the `.env`
file to match, or just pick it from `ui.py`'s dropdown at runtime.

## Things you can try asking

- "What's the correct hook naming convention in Drupal?" → answered from the `drupal` category.
- "How do I prevent SQL injection in a custom module?" → also from the `drupal` category.
- "What's the difference between let and const in JavaScript?" → answered from the `javascript` category.
- "How does the useEffect hook work in React?" → answered from the `react` category.
- "What does box-sizing: border-box do?" → answered from the `html_css` category.
- "How does npm semantic versioning work?" → answered from the `nodejs` category.
- "What's the weather in Ahmedabad right now?" → uses the live weather tool, no documents needed.
- "What's the weather in Ahmedabad, and also how does Drupal handle CSRF protection?" → uses both the tool and document search in one go.
- Ask about a Drupal topic, then follow up with "and what about performance?" → tests whether it remembers the conversation.

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
