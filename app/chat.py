"""Core chat engine: multi-turn conversation, RAG retrieval, tool routing,
and guardrailed structured output — orchestrating a local Ollama model.
"""
import json

import ollama

from app.config import OLLAMA_MODEL, OLLAMA_HOST, MAX_HISTORY_TURNS
from app.retriever import retrieve
from app.tools import get_weather
from app.guardrails import input_guardrail, parse_structured_answer, ChatAnswer
from app.prompt_engineering import (
    INTENT_SYSTEM_PROMPT,
    LOCATION_EXTRACT_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    ANSWER_FEW_SHOT_EXAMPLES,
    GREETING_SYSTEM_PROMPT,
    CATEGORY_PROMPT_STYLES,
    QUICK_GREETING_REPLIES,
)
from app.utils import is_weather_message, is_followup_message, preprocess_query, postprocess_answer

_client = ollama.Client(host=OLLAMA_HOST)


class ChatSession:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.history: list[dict] = []  # [{"role": "user"/"assistant", "content": str}]
        self.model = model

    def set_model(self, model: str) -> None:
        """Switches which Ollama model answers future turns, keeping conversation history intact."""
        self.model = model

    def _classify_intent(self, user_message: str) -> dict:
        if is_weather_message(user_message):
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
            location = self._extract_location(user_message)
            if location is None:
                # The extraction call itself is a small, occasionally-flaky LLM call —
                # seen returning null once for "Do I need an umbrella in London today?"
                # (the assignment's own demo phrasing) even though the same input
                # returned "London" on other calls. Since we already know deterministically
                # that this IS a weather request, one retry is cheap insurance against
                # silently falling through to the RAG path and hallucinating an answer.
                location = self._extract_location(user_message)
            return {"intent": "weather", "location": location}
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
        if not self.history or not is_followup_message(user_message):
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
