"""Basic NLP preprocessing utilities for Cognexia."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def preprocess_text(
    text: str,
    *,
    remove_stopwords: bool = False,
    lowercase: bool = True,
    remove_punctuation: bool = False,
) -> str:
    """Clean and normalize raw text for inference.

    Args:
        text: Raw text input.
        remove_stopwords: If True, remove common stopwords.
        lowercase: If True, lowercase the text.
        remove_punctuation: If True, remove punctuation characters.

    Returns:
        Cleaned and normalized text.
    """

    if not text:
        return ""

    normalized = text.strip()

    if lowercase:
        normalized = normalized.lower()

    if remove_punctuation:
        normalized = _remove_punctuation(normalized)

    normalized = _collapse_whitespace(normalized)

    if remove_stopwords:
        normalized = _remove_stopwords(normalized)

    return normalized


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()


def _remove_punctuation(text: str) -> str:
    cleaned_chars = [
        " " if _is_punctuation(char) else char
        for char in text
    ]
    return "".join(cleaned_chars)


def _is_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P")


def _remove_stopwords(text: str) -> str:
    tokens = [token for token in text.split(" ") if token]
    filtered = [token for token in tokens if token not in DEFAULT_STOPWORDS]
    return " ".join(filtered)


def tokenize(text: str) -> list[str]:
    """Simple tokenizer (space-delimited)."""
    return [token for token in _collapse_whitespace(text).split(" ") if token]
