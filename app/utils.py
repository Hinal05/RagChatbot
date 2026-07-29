"""Query preprocessing, answer postprocessing, and small classification helpers used by the
chat engine — kept separate from app/chat.py's orchestration logic so these pure, easily
unit-testable pieces are easy to find and test independently.
"""
import re

FOLLOW_UP_MAX_WORDS = 8

# A short message only counts as a follow-up (see is_followup_message) if it also contains
# one of these referring pronouns/connectors — otherwise a short, self-contained new
# question (or a short rude/off-topic message) would wrongly inherit the previous topic.
FOLLOW_UP_CUE_RE = re.compile(
    r"\b(it|that|this|those|these|they|them)\b|what about|how about|and what|and how"
)

# Deterministic weather-intent triggers: kept intentionally narrow, avoiding generic words
# like "hot"/"cold"/"rain" that would collide with real web-dev terms (e.g. "hot reloading",
# "cold start"). Each of these is specific enough to reliably mean an actual weather request.
WEATHER_KEYWORDS = ["weather", "umbrella", "forecast", "raining", "snowing", "sunny outside"]


def is_weather_message(user_message: str) -> bool:
    lowered = user_message.lower()
    return any(kw in lowered for kw in WEATHER_KEYWORDS)


def is_followup_message(user_message: str) -> bool:
    """True if a message is short enough AND contains a referring cue to count as a
    follow-up to the prior turn, rather than a self-contained new question."""
    if len(user_message.split()) > FOLLOW_UP_MAX_WORDS:
        return False
    return bool(FOLLOW_UP_CUE_RE.search(user_message.lower()))


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
