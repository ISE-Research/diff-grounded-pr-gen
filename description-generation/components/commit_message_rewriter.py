"""
Builds commit payloads for the LLM to rewrite commit messages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .patch_utils import combine_patches


class CommitMessageRewriter:
    @staticmethod
    def estimate_tokens(text: str) -> int:
        text = text or ""
        return max(1, len(text) // 4)

    @staticmethod
    def trim_payload_by_tokens(
        payload: List[Dict[str, Any]],
        max_tokens: Optional[int],
    ) -> List[Dict[str, Any]]:
        if not max_tokens or max_tokens <= 0:
            return payload
        total = 0
        trimmed: List[Dict[str, Any]] = []
        for item in payload:
            text = f"{item.get('original_message','')}\n{item.get('diff_excerpt','')}"
            cost = CommitMessageRewriter.estimate_tokens(text)
            if trimmed and total + cost > max_tokens:
                break
            total += cost
            trimmed.append(item)
        return trimmed

    @staticmethod
    def build_payload(
        commits: List[Dict[str, Any]],
        use_cmg_commits: bool = False,
        max_lines_per_patch: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for commit in commits:
            sha = commit.get("sha", "")[:40]
            combined_patch = combine_patches(commit.get("patches", []), max_lines_per_patch)
            if use_cmg_commits:
                effective_message = commit.get("cmg_rewritten_message") or commit.get("message", "")
            else:
                effective_message = commit.get("message", "")
            payload.append({
                "sha": sha,
                "author": commit.get("author") or commit.get("author_login"),
                "timestamp": commit.get("timestamp"),
                "original_message": effective_message,
                "diff_excerpt": combined_patch,
            })
        return payload
