"""FastAPI web app: a minimal chat UI backed by the RAG chat engine.

Run: uvicorn main:app --reload --port 8001
Then open http://127.0.0.1:8001
"""
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.chat import ChatSession

app = FastAPI(title="Drupal RAG Chatbot")

SESSION_MAX_AGE_SECONDS = 30 * 60

_sessions: dict[str, ChatSession] = {}
_session_last_used: dict[str, float] = {}


def _prune_stale_sessions() -> None:
    cutoff = time.monotonic() - SESSION_MAX_AGE_SECONDS
    stale = [sid for sid, last_used in _session_last_used.items() if last_used < cutoff]
    for sid in stale:
        _sessions.pop(sid, None)
        _session_last_used.pop(sid, None)


class AskRequest(BaseModel):
    session_id: str | None = None
    message: str


class AskResponse(BaseModel):
    session_id: str
    answer: str
    used_tool: bool
    sources: list[str]


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    _prune_stale_sessions()
    session_id = payload.session_id or str(uuid.uuid4())
    session = _sessions.setdefault(session_id, ChatSession())
    _session_last_used[session_id] = time.monotonic()
    result = session.ask(payload.message)
    return AskResponse(
        session_id=session_id,
        answer=result.answer,
        used_tool=result.used_tool,
        sources=result.sources,
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Drupal RAG Chatbot</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #f4f6fb;
    --panel: #ffffff;
    --border: #e2e6ef;
    --text: #1f2430;
    --muted: #7a8194;
    --primary: #4f5dff;
    --primary-dark: #3a45d1;
    --user-bubble: #4f5dff;
    --user-text: #ffffff;
    --bot-bubble: #eef0f7;
    --bot-text: #1f2430;
  }
  * { box-sizing: border-box; }
  body {
    font-family: 'Inter', system-ui, sans-serif;
    max-width: 720px;
    margin: 40px auto;
    padding: 0 16px;
    background: var(--bg);
    color: var(--text);
  }
  h2 {
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
  }
  .subtitle {
    color: var(--muted);
    font-size: 0.9em;
    margin-bottom: 20px;
  }
  #log {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    height: 460px;
    overflow-y: auto;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(20, 24, 40, 0.04);
  }
  .msg-row { display: flex; margin: 10px 0; }
  .msg-row.user { justify-content: flex-end; }
  .msg-row.bot { justify-content: flex-start; }
  .bubble {
    max-width: 78%;
    padding: 10px 14px;
    border-radius: 16px;
    line-height: 1.45;
    font-size: 0.95em;
    white-space: pre-wrap;
  }
  .msg-row.user .bubble {
    background: var(--user-bubble);
    color: var(--user-text);
    border-bottom-right-radius: 4px;
  }
  .msg-row.bot .bubble {
    background: var(--bot-bubble);
    color: var(--bot-text);
    border-bottom-left-radius: 4px;
  }
  .label {
    display: block;
    font-size: 0.72em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.65;
    margin-bottom: 3px;
  }
  .meta {
    margin-top: 6px;
    font-size: 0.78em;
    opacity: 0.75;
    font-style: italic;
  }
  #input-row { display: flex; gap: 10px; }
  #q {
    flex: 1;
    padding: 12px 14px;
    border: 1px solid var(--border);
    border-radius: 10px;
    font-family: inherit;
    font-size: 0.95em;
    background: var(--panel);
    color: var(--text);
  }
  #q:focus { outline: 2px solid var(--primary); outline-offset: 1px; }
  button {
    padding: 12px 22px;
    border: none;
    border-radius: 10px;
    background: var(--primary);
    color: #fff;
    font-family: inherit;
    font-weight: 600;
    font-size: 0.95em;
    cursor: pointer;
    transition: background 0.15s ease;
  }
  button:hover { background: var(--primary-dark); }
  button:disabled { background: var(--muted); cursor: not-allowed; }
  .typing-dots { display: inline-flex; gap: 4px; padding: 4px 2px; }
  .typing-dots span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--muted);
    animation: bounce 1.2s infinite ease-in-out;
  }
  .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
  .typing-dots span:nth-child(3) { animation-delay: 0.3s; }
  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
    30% { transform: translateY(-4px); opacity: 1; }
  }
</style>
</head>
<body>
  <h2>Drupal RAG Chatbot</h2>
  <div class="subtitle">Ask about Drupal coding standards, security, or performance &mdash; or check the weather.</div>
  <div id="log"></div>
  <div id="input-row">
    <input id="q" placeholder="Ask about Drupal coding standards, security, performance... or weather in a city" />
    <button onclick="send()">Send</button>
  </div>
<script>
let sessionId = null;
async function send() {
  const input = document.getElementById('q');
  const button = document.querySelector('#input-row button');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  appendMsg('user', 'You', text, '');

  input.disabled = true;
  button.disabled = true;
  const typingRow = showTyping();

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId, message: text})
    });
    const data = await res.json();
    sessionId = data.session_id;
    let meta = [];
    if (data.used_tool) meta.push('used external tool');
    if (data.sources && data.sources.length) meta.push('sources: ' + data.sources.join(', '));
    typingRow.remove();
    appendMsg('bot', 'Bot', data.answer, meta.join(' | '));
  } catch (err) {
    typingRow.remove();
    appendMsg('bot', 'Bot', 'Something went wrong reaching the server. Please try again.', '');
  } finally {
    input.disabled = false;
    button.disabled = false;
    input.focus();
  }
}
function showTyping() {
  const log = document.getElementById('log');
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.innerHTML = '<div class="bubble"><span class="label">Bot</span>' +
    '<div class="typing-dots"><span></span><span></span><span></span></div></div>';
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
  return row;
}
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
function appendMsg(role, who, text, meta) {
  const log = document.getElementById('log');
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = '<span class="label">' + who + '</span>' + escapeHtml(text) +
    (meta ? '<div class="meta">' + escapeHtml(meta) + '</div>' : '');
  row.appendChild(bubble);
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}
document.getElementById('q').addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
</script>
</body>
</html>
"""
