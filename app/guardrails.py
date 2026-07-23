"""Guardrails: input scope checks, and a strict schema for the model's final answer.

Two layers:
1. Input guardrail — reject requests that are clearly out of scope or unsafe
   before we spend a model call on them.
2. Output guardrail — force the model to answer in a structured JSON shape
   and validate it; if it doesn't validate, we retry once with a corrective
   instruction rather than silently returning malformed output.
"""
import json
from pydantic import BaseModel, ValidationError

BLOCKED_KEYWORDS = [
    "ignore previous instructions",
    "system prompt",
    "jailbreak",
]


class ChatAnswer(BaseModel):
    answer: str
    used_tool: bool = False
    sources: list[str] = []


def input_guardrail(user_message: str) -> str | None:
    """Returns a rejection reason string if the message should be blocked, else None."""
    lowered = user_message.lower()
    for kw in BLOCKED_KEYWORDS:
        if kw in lowered:
            return "This request attempts to override system instructions and was blocked."
    if len(user_message.strip()) == 0:
        return "Empty message."
    if len(user_message) > 4000:
        return "Message is too long."
    return None


def parse_structured_answer(raw_text: str) -> ChatAnswer | None:
    """Extract and validate a JSON object matching ChatAnswer from raw model output."""
    text = raw_text.strip()
    # Models sometimes wrap JSON in markdown fences — strip those.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start:end + 1])
        return ChatAnswer(**data)
    except (json.JSONDecodeError, ValidationError):
        return None
