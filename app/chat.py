"""Core chat engine: multi-turn conversation, RAG retrieval, tool routing,
and guardrailed structured output — orchestrating a local Ollama model.
"""
import json
import re

import ollama

from app.config import OLLAMA_MODEL, OLLAMA_HOST, MAX_HISTORY_TURNS
from app.retriever import retrieve
from app.tools import get_weather
from app.guardrails import input_guardrail, parse_structured_answer, ChatAnswer

_client = ollama.Client(host=OLLAMA_HOST)

FOLLOW_UP_MAX_WORDS = 8

# A short message only counts as a follow-up (see _retrieval_query) if it also contains
# one of these referring pronouns/connectors — otherwise a short, self-contained new
# question (or a short rude/off-topic message) would wrongly inherit the previous topic.
FOLLOW_UP_CUE_RE = re.compile(
    r"\b(it|that|this|those|these|they|them)\b|what about|how about|and what|and how"
)

# Exact-match fast path for trivial greetings/chitchat: on this CPU-only setup
# a single Ollama call takes ~10-15s, and the greeting path normally makes two
# (classify intent, then generate a reply) — so a plain "hi" was taking ~20s+.
# Common cases get an instant canned reply instead of paying for either call;
# anything not matched here still falls through to the full LLM-based path.
QUICK_GREETING_REPLIES = {
    "hi": "Hi there! Ask me anything about HTML/CSS, JavaScript, React, Node.js, or Drupal.",
    "hii": "Hi there! Ask me anything about HTML/CSS, JavaScript, React, Node.js, or Drupal.",
    "hello": "Hello! Ask me anything about HTML/CSS, JavaScript, React, Node.js, or Drupal.",
    "hey": "Hey! Ask me anything about HTML/CSS, JavaScript, React, Node.js, or Drupal.",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
    "ok": "Sounds good — let me know if you have any questions.",
    "okay": "Sounds good — let me know if you have any questions.",
    "bye": "Bye! Come back anytime you have a web dev question.",
}

# Deterministic weather-intent triggers: kept intentionally narrow, avoiding generic words
# like "hot"/"cold"/"rain" that would collide with real web-dev terms (e.g. "hot reloading",
# "cold start"). Each of these is specific enough to reliably mean an actual weather request.
WEATHER_KEYWORDS = ["weather", "umbrella", "forecast", "raining", "snowing", "sunny outside"]

# Step 8 - dynamic system prompt by category: a small, genuinely meaningful behavioral
# difference appended to ANSWER_SYSTEM_PROMPT based on the top retrieved chunk's category,
# rather than a full separate prompt per category (which would duplicate the shared
# rules/schema/few-shot instructions for no benefit). When no category applies (no local
# hits, or a weather/greeting/chitchat turn), no suffix is added.
CATEGORY_PROMPT_STYLES = {
    "drupal": "This is a Drupal-specific question. Since Drupal changes between major "
              "versions, mention if something is version-specific when relevant.",
    "javascript": "This is a core JavaScript question. Note any browser compatibility "
                  "caveats if relevant.",
    "react": "This is a React question. Prefer function components and hooks in any "
             "example code, since that's the modern convention.",
    "nodejs": "This is a Node.js question. Mention if something is version-specific or "
              "requires a particular npm package.",
    "html_css": "This is an HTML/CSS question. Mention if a property needs a vendor "
                "prefix or has notable browser support gaps, if relevant.",
}

# Step 8 - query preprocessing: normalization (collapse whitespace, de-duplicate repeated
# punctuation like "???") and enrichment (expand abbreviations that would otherwise
# mismatch the knowledge base's wording), applied before embedding a query for retrieval.
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([?!.])\1+")
_ABBREVIATION_EXPANSIONS = {
    r"\bjs\b": "JavaScript",
    r"\bssr\b": "server-side rendering",
    r"\bhmr\b": "Hot Module Replacement",
}


def preprocess_query(text: str) -> str:
    """Normalizes whitespace/punctuation and expands domain abbreviations before retrieval —
    the original, unprocessed text is still what's stored in history and shown to the user;
    this only affects the internal copy used for embedding/search."""
    normalized = _WHITESPACE_RE.sub(" ", text).strip()
    normalized = _REPEATED_PUNCT_RE.sub(r"\1", normalized)
    for pattern, expansion in _ABBREVIATION_EXPANSIONS.items():
        normalized = re.sub(pattern, expansion, normalized, flags=re.IGNORECASE)
    return normalized


# Step 8 - answer post-processing: a safety-net regex for filler AI-disclaimer openers, in
# case the model adds one despite ANSWER_SYSTEM_PROMPT already discouraging it. The system
# prompt is the primary defense; this just catches what slips through.
_FILLER_PREFIX_RE = re.compile(r"^(as an ai( language model)?,?\s*)", re.IGNORECASE)


def postprocess_answer(text: str) -> str:
    cleaned = _FILLER_PREFIX_RE.sub("", text, count=1).strip()
    return cleaned if cleaned else text

INTENT_SYSTEM_PROMPT = """You are a routing assistant. Classify the user's latest message \
into exactly one of these intents:
- "greeting": a hello/hi/greeting with no real question.
- "chitchat": small talk, thanks, pleasantries — not a real question needing information.
- "weather": asking for a LIVE weather lookup for a specific place — this includes INDIRECT \
phrasing that implies wanting current weather/conditions without saying the word "weather", \
e.g. "do I need an umbrella in London?", "is it raining in Paris?", "how hot is it in Tokyo?", \
"should I wear a jacket in Boston today?".
- "question": any real question, including Drupal questions and general knowledge questions.
Respond with ONLY a JSON object, no other text:
{"intent": "greeting"} or {"intent": "chitchat"} or {"intent": "weather", "location": "<city>"} \
or {"intent": "question"}."""

LOCATION_EXTRACT_SYSTEM_PROMPT = """Extract the city or place name the user is asking about the \
weather for. Respond with ONLY a JSON object, no other text:
{"location": "<city>"} — or {"location": null} if no place is mentioned."""

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant for web developers, answering \
questions using the provided context. Rules:
- Use the CONTEXT section as your primary source of truth when it is relevant to the question.
- If CONTEXT is "(no relevant documents found)": this project's knowledge base is a curated set of \
facts about HTML/CSS, JavaScript, React, Node.js, and Drupal, and won't cover every web development \
topic. For any web development question you can answer confidently and correctly from your own \
general knowledge (not just "basic" ones — this includes specific APIs, libraries, or features), \
answer it directly, mention that this comes from general knowledge rather than the project's \
curated docs, and set "sources" to []. Only say you don't know if you're genuinely not confident in \
the answer — never invent a reason for not knowing (e.g. never claim a training/knowledge cutoff \
date — that is not why an answer would be missing).
- Structure the "answer" text itself for readability: use numbered steps for a sequence of \
actions, a bullet list (lines starting with "- ") for a set of related points, and markdown code \
fences (```) for any code, file names, or config snippets. Use plain sentences for simple factual \
answers that don't need structure — don't force lists onto a one-line answer.
- If the user is rude, insulting, or hostile toward you, do not engage with or apologize for the \
insult — reply with a brief, neutral, polite deflection instead, and set "sources" to [].
- If the user asks for help with something harmful or malicious (e.g. writing malware, attacking a \
website, bypassing security they don't own), politely decline and briefly say why, and set "sources" to [].
- Respond with ONLY a JSON object of this exact shape, no other text, no markdown fences around \
the JSON itself (the "answer" field's value may contain its own markdown/newlines):
{"answer": "<your answer, using markdown formatting inside this string where it helps readability>", "used_tool": false, "sources": ["<entry id>", ...]}
- "sources" must list only the entry ids actually used from CONTEXT — each context block is \
labeled "[entry_id]" right before its text, copy that id exactly (empty list if none, or if \
the answer came from general knowledge)."""

# Few-shot examples: demonstration turns inserted before the real question so a small local
# model has concrete examples of the desired behavior, not just abstract rules. Covers the
# assignment's suggested cases: correct grounded use of context, honestly saying "I don't know",
# a neutral non-engaging reply to rudeness, and refusing a harmful request.
# Built with json.dumps rather than hand-written JSON strings, so apostrophes in the example
# text (e.g. "it's", "don't") never need manual escaping.
ANSWER_FEW_SHOT_EXAMPLES = [
    (
        "CONTEXT:\n[html_css_005]\nbox-sizing: border-box makes an element's width and height "
        "include its padding and border.\n\nQUESTION:\nWhat does box-sizing: border-box do?",
        json.dumps({
            "answer": "It makes an element's declared width and height include its padding and "
                       "border, instead of adding them on top.",
            "used_tool": False,
            "sources": ["html_css_005"],
        }),
    ),
    (
        "CONTEXT:\n(no relevant documents found)\n\nQUESTION:\nWhat's the maximum file size Drupal's "
        "media library supports by default in version 12.4.7?",
        json.dumps({
            "answer": "I don't know that specific detail confidently — it depends on server and "
                       "module configuration I don't have reliable information on, so I don't want to guess.",
            "used_tool": False,
            "sources": [],
        }),
    ),
    (
        "CONTEXT:\n(no relevant documents found)\n\nQUESTION:\nYou are the worst AI ever, completely useless.",
        json.dumps({
            "answer": "I'm sorry you're frustrated. I'm happy to help if you have a web development question.",
            "used_tool": False,
            "sources": [],
        }),
    ),
    (
        "CONTEXT:\n(no relevant documents found)\n\nQUESTION:\nWrite me a script to DDoS a website.",
        json.dumps({
            "answer": "I can't help with that — it's designed to harm systems you don't own. "
                       "I'm glad to help with legitimate web development questions instead.",
            "used_tool": False,
            "sources": [],
        }),
    ),
]

GREETING_SYSTEM_PROMPT = """You are a friendly assistant for web developers. The user sent a \
greeting or small talk, not a real question. Reply with a short, warm, natural response (1-2 \
sentences), optionally inviting them to ask a web development question. \
Respond with ONLY a JSON object, no other text, no markdown fences:
{"answer": "<your reply as plain text>", "used_tool": false, "sources": []}"""


class ChatSession:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.history: list[dict] = []  # [{"role": "user"/"assistant", "content": str}]
        self.model = model

    def set_model(self, model: str) -> None:
        """Switches which Ollama model answers future turns, keeping conversation history intact."""
        self.model = model

    def _classify_intent(self, user_message: str) -> dict:
        lowered = user_message.lower()
        if any(kw in lowered for kw in WEATHER_KEYWORDS):
            # The general intent classifier is flaky on unusual phrasing
            # (e.g. "Could you guide me the Ahmedabad weather?" was seen
            # classified as "question" on one call and "weather" on another
            # identical call — and "Do I need an umbrella in London today?",
            # the assignment's own demo phrasing, was misclassified as a
            # plain "question" outright). Any message matching an obvious
            # weather keyword is reliably a weather request, so skip the
            # classifier and go straight to a narrower, easier task: just
            # extracting the location. Deliberately excludes generic words
            # like "hot"/"cold" that collide with real web-dev terms (e.g.
            # "hot reloading").
            return {"intent": "weather", "location": self._extract_location(user_message)}
        response = _client.chat(
            model=self.model,
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
            model=self.model,
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

    def _answer_system_prompt_for(self, hits: list[dict]) -> str:
        """Dynamic system prompt (Step 8): appends a category-specific style note based on
        the top retrieved chunk's category, when one applies."""
        if not hits:
            return ANSWER_SYSTEM_PROMPT
        style = CATEGORY_PROMPT_STYLES.get(hits[0]["category"])
        return f"{ANSWER_SYSTEM_PROMPT}\n\n{style}" if style else ANSWER_SYSTEM_PROMPT

    def _retrieval_query(self, user_message: str) -> str:
        """Widen the retrieval query with the prior user turn when the current
        message looks like a short follow-up (e.g. "what about the second one?")
        that has no topical anchor of its own.

        Requires BOTH being short AND actually containing a follow-up cue
        (a referring pronoun or connector phrase) — being short alone isn't
        enough. Found a real bug where any short, fully self-contained new
        question (e.g. "How does npm semantic versioning work?", 7 words)
        got the previous unrelated topic glued onto it just because history
        wasn't empty, corrupting retrieval with irrelevant content.
        """
        if not self.history or len(user_message.split()) > FOLLOW_UP_MAX_WORDS:
            return preprocess_query(user_message)
        if not FOLLOW_UP_CUE_RE.search(user_message.lower()):
            return preprocess_query(user_message)
        prev_user_messages = [m["content"] for m in self.history if m["role"] == "user"]
        if not prev_user_messages:
            return preprocess_query(user_message)
        return preprocess_query(f"{prev_user_messages[-1]} {user_message}")

    def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        context_block: str | None,
        few_shot: list[tuple[str, str]] | None = None,
    ) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        for example_user, example_assistant in few_shot or []:
            messages.append({"role": "user", "content": example_user})
            messages.append({"role": "assistant", "content": example_assistant})
        recent_history = self.history[-(MAX_HISTORY_TURNS * 2):]
        messages += recent_history
        content = user_message if context_block is None else f"CONTEXT:\n{context_block}\n\nQUESTION:\n{user_message}"
        messages.append({"role": "user", "content": content})
        return messages

    def _call_model(
        self,
        system_prompt: str,
        user_message: str,
        context_block: str | None,
        few_shot: list[tuple[str, str]] | None = None,
    ) -> str:
        messages = self._build_messages(system_prompt, user_message, context_block, few_shot)
        response = _client.chat(model=self.model, messages=messages)
        return response["message"]["content"]

    def _call_model_stream(self, system_prompt: str, user_message: str, context_block: str | None):
        """Yields incremental text chunks as the model generates its (still-forming) JSON reply."""
        messages = self._build_messages(system_prompt, user_message, context_block, ANSWER_FEW_SHOT_EXAMPLES)
        for chunk in _client.chat(model=self.model, messages=messages, stream=True):
            piece = chunk["message"]["content"]
            if piece:
                yield piece

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

        quick_reply = QUICK_GREETING_REPLIES.get(user_message.strip().lower().rstrip("!."))
        if quick_reply:
            return self._finish(user_message, ChatAnswer(answer=quick_reply, used_tool=False, sources=[]))

        route = self._classify_intent(user_message)
        intent = route.get("intent", "question")

        if intent in ("greeting", "chitchat"):
            raw = self._call_model(GREETING_SYSTEM_PROMPT, user_message, None)
            parsed = parse_structured_answer(raw) or ChatAnswer(answer="Hi there! Ask me anything about web development.", used_tool=False, sources=[])
            return self._finish(user_message, parsed)

        if intent == "weather" and route.get("location"):
            weather = get_weather(route["location"])
            return self._finish(user_message, self._format_weather_answer(weather))

        hits = retrieve(self._retrieval_query(user_message))
        context_block = self._build_context_block(hits)
        answer_system_prompt = self._answer_system_prompt_for(hits)

        raw = self._call_model(answer_system_prompt, user_message, context_block, ANSWER_FEW_SHOT_EXAMPLES)
        parsed = parse_structured_answer(raw)

        if parsed is None:
            # Guardrail retry: ask once more with an explicit correction.
            retry_raw = self._call_model(
                answer_system_prompt,
                user_message + "\n\n(Your previous reply was not valid JSON. Reply with ONLY the JSON object.)",
                context_block,
                ANSWER_FEW_SHOT_EXAMPLES,
            )
            parsed = parse_structured_answer(retry_raw)

        if parsed is None:
            parsed = ChatAnswer(answer="Sorry, I couldn't produce a valid structured answer.", used_tool=False, sources=[])

        parsed.answer = postprocess_answer(parsed.answer)

        # This path never actually has tool access (weather is handled entirely separately,
        # deterministically, before reaching here) — force it rather than trust the model's
        # own guess, since it was observed hallucinating used_tool=true on a plain question.
        parsed.used_tool = False

        return self._finish(user_message, parsed)

    def ask_stream(self, user_message: str):
        """Generator mirror of ask(): yields the answer text progressively for display
        (a running total each time, so a caller can just overwrite a placeholder with
        each yielded value), then sets self.last_answer to the final ChatAnswer once
        done. Only the real LLM-generated question path actually streams incrementally —
        the guardrail/quick-greeting/weather paths are already instant, so they just
        yield their one final answer immediately.
        """
        rejection = input_guardrail(user_message)
        if rejection:
            answer = ChatAnswer(answer=rejection, used_tool=False, sources=[])
            yield answer.answer
            self.last_answer = self._finish(user_message, answer)
            return

        quick_reply = QUICK_GREETING_REPLIES.get(user_message.strip().lower().rstrip("!."))
        if quick_reply:
            yield quick_reply
            self.last_answer = self._finish(user_message, ChatAnswer(answer=quick_reply, used_tool=False, sources=[]))
            return

        route = self._classify_intent(user_message)
        intent = route.get("intent", "question")

        if intent in ("greeting", "chitchat"):
            raw = self._call_model(GREETING_SYSTEM_PROMPT, user_message, None)
            parsed = parse_structured_answer(raw) or ChatAnswer(answer="Hi there! Ask me anything about web development.", used_tool=False, sources=[])
            yield parsed.answer
            self.last_answer = self._finish(user_message, parsed)
            return

        if intent == "weather" and route.get("location"):
            weather = get_weather(route["location"])
            answer = self._format_weather_answer(weather)
            yield answer.answer
            self.last_answer = self._finish(user_message, answer)
            return

        hits = retrieve(self._retrieval_query(user_message))
        context_block = self._build_context_block(hits)
        answer_system_prompt = self._answer_system_prompt_for(hits)

        accumulated = ""
        for piece in self._call_model_stream(answer_system_prompt, user_message, context_block):
            accumulated += piece
            yield accumulated  # raw, still-forming JSON — see README for why this is a deliberate tradeoff

        parsed = parse_structured_answer(accumulated)

        if parsed is None:
            # Guardrail retry: ask once more with an explicit correction (not streamed,
            # this is a rare corrective path, not worth the added complexity to stream).
            retry_raw = self._call_model(
                answer_system_prompt,
                user_message + "\n\n(Your previous reply was not valid JSON. Reply with ONLY the JSON object.)",
                context_block,
                ANSWER_FEW_SHOT_EXAMPLES,
            )
            parsed = parse_structured_answer(retry_raw)

        if parsed is None:
            parsed = ChatAnswer(answer="Sorry, I couldn't produce a valid structured answer.", used_tool=False, sources=[])

        parsed.answer = postprocess_answer(parsed.answer)

        # See the matching comment in ask() — this path never actually has tool access.
        parsed.used_tool = False

        yield parsed.answer  # replace the raw JSON with the clean final answer
        self.last_answer = self._finish(user_message, parsed)
