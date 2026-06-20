"""
Extracts anchor terms from diff excerpts for grounding.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable, List


IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
PATHLIKE_RE = re.compile(r"\b[A-Za-z0-9_./-]{4,}\b")

STOPWORDS = {
    "function", "return", "const", "let", "var", "class", "public", "private",
    "protected", "static", "final", "import", "from", "export", "def", "self",
    "true", "false", "null", "none", "void", "string", "int", "float", "bool",
    "if", "else", "for", "while", "switch", "case", "break", "continue",
    "new", "this", "super", "try", "catch", "finally", "raise", "throws",
    "await", "async", "yield", "with", "and", "or", "not",
}

NOISE_TOKENS = {
    "repo", "repo_name", "pr", "pr_number", "commit", "commits", "file", "files",
    "print", "str", "the", "this", "that", "these", "those", "data", "result",
}


def _is_camel_case(token: str) -> bool:
    return any(c.islower() for c in token) and any(c.isupper() for c in token)


def _is_useful_token(token: str) -> bool:
    lower = token.lower()
    if lower in STOPWORDS or lower in NOISE_TOKENS:
        return False
    if len(token) < 4:
        return False
    if _is_camel_case(token):
        return True
    if any(ch.isdigit() for ch in token):
        return True
    if "_" in token:
        parts = [p for p in token.split("_") if len(p) >= 3 and p.lower() not in STOPWORDS]
        return len(parts) >= 1
    return False


def extract_anchor_terms(texts: Iterable[str], max_terms: int) -> List[str]:
    if not max_terms or max_terms <= 0:
        return []
    counter: Counter[str] = Counter()
    for text in texts:
        if not text:
            continue
        for token in IDENTIFIER_RE.findall(text):
            if _is_useful_token(token):
                counter[token] += 1
        for token in PATHLIKE_RE.findall(text):
            if "/" in token or "." in token:
                if token.lower() in STOPWORDS or token.lower() in NOISE_TOKENS:
                    continue
                if len(token) < 4:
                    continue
                counter[token] += 1
    if not counter:
        return []
    ranked = sorted(counter.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
    return [token for token, _ in ranked[:max_terms]]
