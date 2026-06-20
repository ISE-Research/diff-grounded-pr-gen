"""
Shared helpers for trimming and combining patches before sending them to the LLM.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def clean_patch(patch: str, max_lines: Optional[int]) -> str:
    """Strip git headers and optionally truncate to max_lines."""
    patch = re.sub(r'^(diff --git a/.*? b/.*?)$', '', patch, flags=re.MULTILINE)
    patch = re.sub(r'^(index .*|--- a/.*|\+\+\+ b/.*)$', '', patch, flags=re.MULTILINE)
    lines = [
        line.rstrip()
        for line in patch.splitlines()
        if line.strip()
    ]
    if max_lines is not None:
        return "\n".join(lines[: max_lines])
    return "\n".join(lines)


def combine_patches(patches: List[Dict[str, Any]], max_lines: Optional[int]) -> str:
    """Concatenate individual file patches for a commit and optionally truncate."""
    snippets: List[str] = []
    for patch in patches or []:
        text = patch.get("patch")
        if not text:
            continue
        cleaned = clean_patch(text, max_lines)
        if cleaned:
            snippets.append(cleaned)
    if not snippets:
        return ""
    combined = "\n".join(snippets)
    lines = combined.splitlines()
    if max_lines is not None:
        return "\n".join(lines[: max_lines])
    return "\n".join(lines)
