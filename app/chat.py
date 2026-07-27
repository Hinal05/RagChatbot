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

LOCATION_EXTRACT_SYSTEM_PROMPT = """Extract the city or place name the user is asking about the \
weather for. Respond with ONLY a JSON object, no other text:
{"location": "<city>"} — or {"location": null} if no place is mentioned."""

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant for Drupal developers, answering \
questions using the provided context. Rules:
- Use the CONTEXT section as your primary source of truth when it is relevant to the question.
- If CONTEXT is "(no relevant documents found)": this project's knowledge base is small (a few \
short docs on coding standards, security, performance, and module development) and won't cover most \
Drupal topics. For any Drupal-related question you can answer confidently and correctly from your \
own general knowledge (not just "basic" ones — this includes specific APIs, modules, or features), \
answer it directly, mention that this comes from general knowledge rather than the project's \
curated docs, and set "sources" to []. Only say you don't know if you're genuinely not confident in \
the answer — never invent a reason for not knowing (e.g. never claim a training/knowledge cutoff \
date — that is not why an answer would be missing).
- Structure the "answer" text itself for readability: use numbered steps for a sequence of \
actions, a bullet list (lines starting with "- ") for a set of related points, and markdown code \
fences (```) for any code, file names, or config snippets. Use plain sentences for simple factual \
answers that don't need structure — don't force lists onto a one-line answer.
- Respond with ONLY a JSON object of this exact shape, no other text, no markdown fences around \
the JSON itself (the "answer" field's value may contain its own markdown/newlines):
{"answer": "<your answer, using markdown formatting inside this string where it helps readability>", "used_tool": false, "sources": ["<entry id>", ...]}
- "sources" must list only the entry ids actually used from CONTEXT — each context block is \
labeled "[entry_id]" right before its text, copy that id exactly (empty list if none, or if \
the answer came from general knowledge)."""

GREETING_SYSTEM_PROMPT = """You are a friendly assistant for Drupal developers. The user sent a \
greeting or small talk, not a real question. Reply with a short, warm, natural response (1-2 \
sentences), optionally inviting them to ask a Drupal question. \
Respond with ONLY a JSON object, no other text, no markdown fences:
{"answer": "<your reply as plain text>", "used_tool": false, "sources": []}"""


class ChatSession:
    def __init__(self):
        self.history: list[dict] = []  # [{"role": "user"/"assistant", "content": str}]

    def _classify_intent(self, user_message: str) -> dict:
        if "weather" in user_message.lower():
            # The general intent classifier is flaky on unusual phrasing
            # (e.g. "Could you guide me the Ahmedabad weather?" was seen
            # classified as "question" on one call and "weather" on another
            # identical call). Any message mentioning "weather" is reliably
            # a weather request, so skip the classifier and go straight to
            # a narrower, easier task: just extracting the location.
            return {"intent": "weather", "location": self._extract_location(user_message)}
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

    def _extract_location(self, user_message: str) -> str | None:
        response = _client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": LOCATION_EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        raw = response["message"]["content"].strip()
        try:
            start, end = raw.find("{"), raw.rfind("}")
            return json.loads(raw[start:end + 1]).get("location")
        except (json.JSONDecodeError, ValueError):
            return None

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

    def _call_model(self, system_prompt: str, user_message: str, context_block: str | None) -> str:
        recent_history = self.history[-(MAX_HISTORY_TURNS * 2):]
        messages = [{"role": "system", "content": system_prompt}]
        messages += recent_history
        content = user_message if context_block is None else f"CONTEXT:\n{context_block}\n\nQUESTION:\n{user_message}"
        messages.append({"role": "user", "content": content})
        response = _client.chat(model=OLLAMA_MODEL, messages=messages)
        return response["message"]["content"]

    def _format_weather_answer(self, weather: dict) -> ChatAnswer:
        """Builds the weather reply directly from the tool's structured data instead of
        letting the LLM paraphrase it — avoids the model turning e.g. "thunderstorm"
        into odd wording like "thunderous" when rendering the numbers into prose."""
        if "error" in weather:
            return ChatAnswer(answer=weather["error"], used_tool=True, sources=[])
        answer = (
            f"Current weather in {weather['location']}: {weather['condition']}, "
            f"{weather['temperature_c']}°C (feels like {weather['feels_like_c']}°C), "
            f"humidity {weather['humidity_pct']}%, wind {weather['windspeed_kmh']} km/h."
        )
        return ChatAnswer(answer=answer, used_tool=True, sources=[])

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
            raw = self._call_model(GREETING_SYSTEM_PROMPT, user_message, None)
            parsed = parse_structured_answer(raw) or ChatAnswer(answer="Hi there! Ask me anything about Drupal.", used_tool=False, sources=[])
            return self._finish(user_message, parsed)

        if intent == "weather" and route.get("location"):
            weather = get_weather(route["location"])
            return self._finish(user_message, self._format_weather_answer(weather))

        hits = retrieve(self._retrieval_query(user_message))
        context_block = self._build_context_block(hits)

        raw = self._call_model(ANSWER_SYSTEM_PROMPT, user_message, context_block)
        parsed = parse_structured_answer(raw)

        if parsed is None:
            # Guardrail retry: ask once more with an explicit correction.
            retry_raw = self._call_model(
                ANSWER_SYSTEM_PROMPT,
                user_message + "\n\n(Your previous reply was not valid JSON. Reply with ONLY the JSON object.)",
                context_block,
            )
            parsed = parse_structured_answer(retry_raw)

        if parsed is None:
            parsed = ChatAnswer(answer="Sorry, I couldn't produce a valid structured answer.", used_tool=False, sources=[])

        return self._finish(user_message, parsed)
