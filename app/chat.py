"""Core chat engine: multi-turn conversation, RAG retrieval, tool routing,
and guardrailed structured output — orchestrating a local Ollama model.
"""
import json

import ollama

from app.config import OLLAMA_MODEL, OLLAMA_HOST, MAX_HISTORY_TURNS
from app.retriever import retrieve
from app.tools import get_weather
from app.guardrails import input_guardrail, parse_structured_answer, ChatAnswer

_client = ollama.Client(host=OLLAMA_HOST)

FOLLOW_UP_MAX_WORDS = 8

INTENT_SYSTEM_PROMPT = """You are a routing assistant. Classify the user's latest message \
into exactly one of these intents:
- "greeting": a hello/hi/greeting with no real question.
- "chitchat": small talk, thanks, pleasantries — not a real question needing information.
- "weather": asking for a LIVE weather lookup for a specific place.
- "question": any real question, including Drupal questions and general knowledge questions.
Respond with ONLY a JSON object, no other text:
{"intent": "greeting"} or {"intent": "chitchat"} or {"intent": "weather", "location": "<city>"} \
or {"intent": "question"}."""

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant for Drupal developers, answering \
questions using the provided context. Rules:
- Use the CONTEXT section as your primary source of truth when it is relevant to the question.
- If CONTEXT is "(no relevant documents found)":
  - If the question is a basic general-knowledge question (e.g. "What is Drupal?") that you can \
answer confidently and correctly from your own knowledge, answer it directly and set "sources" to [].
  - If the question depends on this project's specific knowledge base (its coding standards, \
security practices, performance tuning, or module development conventions) and you don't have that \
context, say plainly that you don't have that specific information in your knowledge base. Do not \
invent a reason (e.g. never claim a training/knowledge cutoff date — that is not why the answer is \
missing).
- If TOOL_RESULT is provided, incorporate it directly into your answer.
- Respond with ONLY a JSON object of this exact shape, no other text, no markdown fences:
{"answer": "<your answer as plain text>", "used_tool": <true or false>, "sources": ["<filename>", ...]}
- "sources" must list only the filenames actually used from CONTEXT (empty list if none, or if \
the answer came purely from TOOL_RESULT or general knowledge)."""

GREETING_SYSTEM_PROMPT = """You are a friendly assistant for Drupal developers. The user sent a \
greeting or small talk, not a real question. Reply with a short, warm, natural response (1-2 \
sentences), optionally inviting them to ask a Drupal question. \
Respond with ONLY a JSON object, no other text, no markdown fences:
{"answer": "<your reply as plain text>", "used_tool": false, "sources": []}"""


class ChatSession:
    def __init__(self):
        self.history: list[dict] = []  # [{"role": "user"/"assistant", "content": str}]

    def _classify_intent(self, user_message: str) -> dict:
        response = _client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        raw = response["message"]["content"].strip()
        try:
            start, end = raw.find("{"), raw.rfind("}")
            return json.loads(raw[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return {"intent": "question"}

    def _build_context_block(self, hits: list[dict]) -> str:
        if not hits:
            return "(no relevant documents found)"
        return "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)

    def _retrieval_query(self, user_message: str) -> str:
        """Widen the retrieval query with the prior user turn when the current
        message looks like a short follow-up (e.g. "what about the second one?")
        that has no topical anchor of its own."""
        if not self.history or len(user_message.split()) > FOLLOW_UP_MAX_WORDS:
            return user_message
        prev_user_messages = [m["content"] for m in self.history if m["role"] == "user"]
        if not prev_user_messages:
            return user_message
        return f"{prev_user_messages[-1]} {user_message}"

    def _call_model(self, system_prompt: str, user_message: str, context_block: str | None, tool_result: dict | None) -> str:
        recent_history = self.history[-(MAX_HISTORY_TURNS * 2):]
        messages = [{"role": "system", "content": system_prompt}]
        messages += recent_history
        if context_block is None and tool_result is None:
            content = user_message
        else:
            tool_block = json.dumps(tool_result) if tool_result else "(none)"
            content = f"CONTEXT:\n{context_block}\n\nTOOL_RESULT:\n{tool_block}\n\nQUESTION:\n{user_message}"
        messages.append({"role": "user", "content": content})
        response = _client.chat(model=OLLAMA_MODEL, messages=messages)
        return response["message"]["content"]

    def _finish(self, user_message: str, parsed: ChatAnswer) -> ChatAnswer:
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": parsed.answer})
        return parsed

    def ask(self, user_message: str) -> ChatAnswer:
        rejection = input_guardrail(user_message)
        if rejection:
            return ChatAnswer(answer=rejection, used_tool=False, sources=[])

        route = self._classify_intent(user_message)
        intent = route.get("intent", "question")

        if intent in ("greeting", "chitchat"):
            raw = self._call_model(GREETING_SYSTEM_PROMPT, user_message, None, None)
            parsed = parse_structured_answer(raw) or ChatAnswer(answer="Hi there! Ask me anything about Drupal.", used_tool=False, sources=[])
            return self._finish(user_message, parsed)

        tool_result = None
        context_block = "(no relevant documents found)"
        if intent == "weather" and route.get("location"):
            tool_result = get_weather(route["location"])
        else:
            hits = retrieve(self._retrieval_query(user_message))
            context_block = self._build_context_block(hits)

        raw = self._call_model(ANSWER_SYSTEM_PROMPT, user_message, context_block, tool_result)
        parsed = parse_structured_answer(raw)

        if parsed is None:
            # Guardrail retry: ask once more with an explicit correction.
            retry_raw = self._call_model(
                ANSWER_SYSTEM_PROMPT,
                user_message + "\n\n(Your previous reply was not valid JSON. Reply with ONLY the JSON object.)",
                context_block,
                tool_result,
            )
            parsed = parse_structured_answer(retry_raw)

        if parsed is None:
            parsed = ChatAnswer(answer="Sorry, I couldn't produce a valid structured answer.", used_tool=False, sources=[])

        return self._finish(user_message, parsed)
