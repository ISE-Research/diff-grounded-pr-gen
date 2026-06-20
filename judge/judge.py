#!/usr/bin/env python3
"""
LLM-based judge that compares generated vs. original PR descriptions using graph context.

Reads a descriptions file (descriptions-*.json) from results/pr-description/<provider>/ and the knowledge
graph at results/knowledge_graph/graph-<dataset>.json, then scores each PR 1-5 with a preference.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
DATA_COLLECTION_DIR = ROOT_DIR / "data-collection"
PR_DIR = ROOT_DIR / "description-generation"
for path in (ROOT_DIR, DATA_COLLECTION_DIR, PR_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from dotenv import load_dotenv

from config.loader import load_pipeline_config
from knowledge_graph import KnowledgeGraphReader
from components.file_diff_summarizer import (
    FileDiffSummarizer,
    _count_public_symbols,
    _is_docs_file,
    _language_for_file,
)
from components.ranking import compute_file_scores, rank_commits
from components.commit_message_rewriter import CommitMessageRewriter
from components.patch_utils import clean_patch
from wrappers.mistral.llm_client import LLMClientWrapper as MistralClient
from wrappers.openai.llm_client import LLMClientWrapper as OpenAIClient
from wrappers.llama.llm_client import LLMClientWrapper as LlamaClient
from wrappers.deepseek.llm_client import LLMClientWrapper as DeepSeekClient
from wrappers.gemini.llm_client import LLMClientWrapper as GeminiClient


def _find_latest_descriptions_file(results_dir: Path, dataset_name: str | None = None) -> Path:
    if dataset_name:
        candidates = list(results_dir.glob(f"descriptions-{dataset_name}-*.json"))
    else:
        candidates = list(results_dir.glob("descriptions-*.json"))
    if not candidates:
        raise FileNotFoundError(f"No descriptions JSON found in {results_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_latest_judge_file(judge_dir: Path, descriptions_stem: str) -> Path | None:
    candidates = list(judge_dir.glob(f"{descriptions_stem}-judge-*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _next_judgments_file(dir_path: Path) -> Path:
    pattern = re.compile(r"judgments-(\d+)\.json$")
    max_idx = 0
    for path in dir_path.glob("judgments-*.json"):
        match = pattern.match(path.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return dir_path / f"judgments-{max_idx + 1}.json"


SMALL_ITEM_LIMIT = 15
TOP_K_ITEMS = 25
MAX_DIFF_LINES = 50
DEFAULT_MAX_FILE_LIST_ITEMS = 200
DEFAULT_MAX_PROMPT_TOKENS = 1500000
DEFAULT_PROMPT_TOKEN_SAFETY_FACTOR = 2.0
DEFAULT_MAX_SECONDS_PER_PR = 1800


def _estimate_tokens(text: str) -> int:
    return CommitMessageRewriter.estimate_tokens(text or "")


def _estimate_prompt_tokens(system_prompt: str, user_prompt: str) -> int:
    text = (system_prompt or "") + "\n" + (user_prompt or "")
    # Conservative estimate: ~3 chars per token.
    return max(1, len(text) // 3)


def _safe_parse_json(raw: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _truncate_lines(text: str, max_lines: int) -> str:
    lines = (text or "").splitlines()
    if max_lines is None or len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines])


def _build_file_list(
    files: List[Dict[str, Any]],
    max_items: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    items: List[Dict[str, Any]] = []
    for file in files or []:
        filename = file.get("filename") or ""
        if not filename:
            continue
        items.append(
            {
                "filename": filename,
                "status": file.get("status"),
                "additions": file.get("additions"),
                "deletions": file.get("deletions"),
                "is_docs": _is_docs_file(filename),
            }
        )
    truncated = False
    if max_items and len(items) > max_items:
        items = items[:max_items]
        truncated = True
    return items, truncated


def _ensure_file_list_meta(
    file_list: List[Dict[str, Any]],
    pr_context: Dict[str, Any],
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    meta = meta or {}
    total_files = meta.get("total_files")
    returned_files = meta.get("returned_files")
    truncated = meta.get("truncated")
    if total_files is None:
        total_files = len(pr_context.get("files", []) or [])
    if returned_files is None:
        returned_files = len(file_list or [])
    if truncated is None:
        truncated = returned_files < total_files
    return {
        "total_files": total_files,
        "returned_files": returned_files,
        "truncated": bool(truncated),
    }




def _only_changed_lines(patch: str) -> str:
    out: List[str] = []
    for line in (patch or "").splitlines():
        if line.startswith(("diff --git", "@@", "+++", "---")):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            out.append(line)
        elif line.startswith("-") and not line.startswith("---"):
            out.append(line)
    return "\n".join(out)


def _build_commit_diff_excerpts(commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for commit in commits or []:
        sha = commit.get("sha")
        patches = commit.get("patches") or []
        diff_lines: List[str] = []
        has_docs = False
        has_public_change = False
        for p in patches:
            filename = p.get("filename") or ""
            if filename:
                diff_lines.append(f"<FILE> {filename}")
                if _is_docs_file(filename):
                    has_docs = True
            patch = p.get("patch") or ""
            language = _language_for_file(filename)
            public_added, public_removed = _count_public_symbols(patch, language)
            if public_added or public_removed:
                has_public_change = True
            cleaned = clean_patch(patch, None)
            changed = _only_changed_lines(cleaned)
            if changed:
                diff_lines.extend(changed.splitlines())
        diff_text = "\n".join(diff_lines)
        diff_text = _truncate_lines(diff_text, MAX_DIFF_LINES)
        diff_len = len(diff_text.splitlines())
        scored.append(
            {
                "sha": sha,
                "message": commit.get("message"),
                "diff_excerpt": diff_text,
                "diff_lines": diff_len,
                "has_docs": has_docs,
                "has_public_change": has_public_change,
            }
        )

    if len(scored) <= SMALL_ITEM_LIMIT:
        return scored

    docs_commits = [c for c in scored if c.get("has_docs")]
    important_commits = [
        c
        for c in scored
        if c.get("has_public_change") and c not in docs_commits
    ]
    remaining = [c for c in scored if c not in docs_commits and c not in important_commits]

    remaining.sort(key=lambda c: c.get("diff_lines", 0), reverse=True)
    picked = docs_commits + important_commits
    if len(picked) < TOP_K_ITEMS:
        picked.extend(remaining[: max(0, TOP_K_ITEMS - len(picked))])
    else:
        picked = picked[:TOP_K_ITEMS]

    picked.sort(key=lambda c: c.get("diff_lines", 0), reverse=True)
    return picked


def _build_file_diff_excerpts(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = FileDiffSummarizer.build_payload(files, max_lines_per_patch=None)
    cleaned: List[Dict[str, Any]] = []
    for item in payload:
        diff = _only_changed_lines(item.get("diff_excerpt") or "")
        diff = _truncate_lines(diff, MAX_DIFF_LINES)
        cleaned.append(
            {
                "filename": item.get("filename"),
                "status": item.get("status"),
                "additions": item.get("additions"),
                "deletions": item.get("deletions"),
                "diff_excerpt": diff,
                "diff_lines": len(diff.splitlines()),
            }
        )
    return cleaned


def _build_file_list_item(diff_item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "filename": diff_item.get("filename"),
        "status": diff_item.get("status"),
        "additions": diff_item.get("additions"),
        "deletions": diff_item.get("deletions"),
    }


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    max_chars = max(1, max_tokens * 4)  # heuristic: ~4 chars per token
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars]


def _chunk_evidence_by_tokens(
    commit_payload: List[Dict[str, Any]],
    file_summary_diffs: List[Dict[str, Any]],
    commit_messages: List[Dict[str, Any]],
    file_summaries: List[Dict[str, Any]],
    linked_issues: List[Dict[str, Any]],
    description: str,
    total_files: int,
    max_prompt_tokens: int,
) -> List[Dict[str, Any]]:
    # Build a conservative base prompt estimate without file diffs/list.
    base_sys, base_user = _build_full_judgment_single_prompt(
        commit_payload,
        [],
        [],
        {"total_files": total_files, "returned_files": 0, "truncated": True},
        commit_messages,
        file_summaries,
        linked_issues,
        description,
    )
    base_tokens = _estimate_prompt_tokens(base_sys, base_user)

    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_tokens = base_tokens

    for item in file_summary_diffs:
        item_tokens = _estimate_tokens(json.dumps(item, indent=2)) + _estimate_tokens(
            json.dumps(_build_file_list_item(item), indent=2)
        )
        if current and current_tokens + item_tokens > max_prompt_tokens:
            chunks.append(current)
            current = []
            current_tokens = base_tokens
        current.append(item)
        current_tokens += item_tokens

    if current:
        chunks.append(current)

    fixed_chunks: List[Dict[str, Any]] = []
    for chunk in chunks:
        file_list = [_build_file_list_item(i) for i in chunk]
        file_list_meta = {
            "total_files": total_files,
            "returned_files": len(file_list),
            "truncated": len(file_list) < total_files,
        }
        sys_prompt, user_prompt = _build_full_judgment_single_prompt(
            commit_payload,
            chunk,
            file_list,
            file_list_meta,
            commit_messages,
            file_summaries,
            linked_issues,
            description,
        )
        if _estimate_prompt_tokens(sys_prompt, user_prompt) <= max_prompt_tokens:
            fixed_chunks.append(
                {
                    "file_summary_diffs": chunk,
                    "file_list": file_list,
                    "file_list_meta": file_list_meta,
                }
            )
            continue

        # Truncate the largest diff excerpts in this chunk.
        shrunk = []
        for item in chunk:
            cloned = dict(item)
            diff = cloned.get("diff_excerpt") or ""
            cloned["diff_excerpt"] = _truncate_text_to_tokens(diff, max(1, max_prompt_tokens // 8))
            shrunk.append(cloned)

        file_list = [_build_file_list_item(i) for i in shrunk]
        file_list_meta = {
            "total_files": total_files,
            "returned_files": len(file_list),
            "truncated": len(file_list) < total_files,
        }
        fixed_chunks.append(
            {
                "file_summary_diffs": shrunk,
                "file_list": file_list,
                "file_list_meta": file_list_meta,
            }
        )

    return fixed_chunks


def _primary_reason_from_breakdown(breakdown: Dict[str, Any]) -> str | None:
    if not isinstance(breakdown, dict):
        return None
    penalties = {
        "correctness": float(breakdown.get("correctness_penalty", 0.0) or 0.0),
        "coverage": float(breakdown.get("coverage_penalty", 0.0) or 0.0),
        "clarity": float(breakdown.get("clarity_penalty", 0.0) or 0.0),
    }
    # Pick the largest penalty; if all are zero, return None.
    primary = max(penalties, key=lambda k: penalties[k])
    return primary if penalties[primary] > 0 else None


def _rubric_spec() -> Dict[str, Any]:
    return {
        "scoring": "Start from 5.0, subtract penalties. Scores can be in 0.5 increments (e.g., 4.5).",
        "correctness": {
            "max_penalty": 2.0,
            "per_issue": 0.5,
            "cap_rule": "3+ unsupported claims -> 2.0",
        },
        "coverage": {
            "max_penalty": 2.0,
            "per_issue": 0.5,
            "cap_rule": "3+ missing key changes -> 2.0",
        },
        "clarity": {
            "max_penalty": 1.0,
            "per_issue": 0.5,
            "cap_rule": "multiple clarity issues -> 1.0",
        },
    }


def _normalize_penalty(value: float | None) -> float:
    try:
        num = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return abs(num)


def _score_from_penalties(correctness: float, coverage: float, clarity: float) -> float:
    c = _normalize_penalty(correctness)
    v = _normalize_penalty(coverage)
    l = _normalize_penalty(clarity)
    total = 5.0 - c - v - l
    total = max(1.0, min(5.0, total))
    return round(total, 1)


def _build_full_judgment_single_prompt(
    commit_payload: List[Dict[str, Any]],
    file_summary_diffs: List[Dict[str, Any]],
    file_list: List[Dict[str, Any]],
    file_list_meta: Dict[str, Any],
    commit_messages: List[Dict[str, Any]],
    file_summaries: List[Dict[str, Any]],
    linked_issues: List[Dict[str, Any]],
    description: str,
) -> Tuple[str, str]:
    system_prompt = (
        "You are a senior software engineer and strict reviewer. "
        "Evaluate correctness, coverage, and clarity as separate axes. "
        "Use only the provided evidence; do not infer. "
        "Do not use external knowledge or cross-reference anything outside the payload."
    )
    user_prompt = (
        "Commit payload (messages + diff excerpts):\n"
        f"{json.dumps(commit_payload, indent=2)}\n\n"
        "File summary diffs (summaries + diff excerpts):\n"
        f"{json.dumps(file_summary_diffs, indent=2)}\n\n"
        "File list (names + metadata, existence only):\n"
        f"{json.dumps(file_list, indent=2)}\n"
        f"File list meta: {json.dumps(file_list_meta, indent=2)}\n\n"
        "Commit messages:\n"
        f"{json.dumps(commit_messages, indent=2)}\n\n"
        "File summaries:\n"
        f"{json.dumps(file_summaries, indent=2)}\n\n"
        "Linked issues:\n"
        f"{json.dumps(linked_issues, indent=2)}\n\n"
        "Description to evaluate:\n"
        f"{description.strip()}\n\n"
        "Rules:\n"
        "- Judge correctness, coverage, and clarity independently; do not let one influence another.\n"
        "- Use only the evidence above; do not infer facts or rely on external knowledge.\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "correctness_penalty": number,\n'
        '  "coverage_penalty": number,\n'
        '  "clarity_penalty": number,\n'
        '  "correctness_penalties": ["unsupported claim ... (evidence: sha/filename)"],\n'
        '  "coverage_penalties": ["missing key change ... (evidence: sha/filename)"],\n'
        '  "clarity_penalties": ["clarity issue ... (evidence: quote)"]\n'
        "}\n"
        "Correctness penalties: 0.5 per unsupported/incorrect claim, cap at 2.0.\n"
        "Coverage penalties: 0.5 per missing key change, cap at 2.0.\n"
        "Clarity penalties: 0.5 per clarity issue, cap at 1.0.\n"
        "Return non-negative penalty values (do not use negative numbers).\n"
        "A claim is supported if it is grounded in a diff excerpt OR explicitly stated in a linked issue.\n"
        "File list evidence confirms a file exists/changed but does not prove behavior or content.\n"
        "- Do NOT penalize missing documentation-only changes; treat doc/template changes as groupable.\n"
        "- If the description groups repeated template/doc changes (e.g., 'scaffolded product templates'), accept it.\n"
        "- High-level summaries are acceptable if they align with the evidence; do not require every minor change.\n"
        "- Each Key Changes and Notable Changes bullet must include at least one concrete identifier/token from the evidence (diff excerpts or commit messages).\n"
        "- If there is at least one non-doc code file change, the Summary must include at least one identifier/token from the evidence (diff excerpts or commit messages).\n"
        "- If multiple files/commits changed, require coverage of at least two distinct evidence items.\n"
        "- Identify the top 5 non-doc files by (additions + deletions). If the description fails to mention any of them, apply a coverage penalty.\n"
        "- Treat non-doc files as files not under docs/ and without extensions: .md, .txt, .rst.\n"
        "- An identifier/token can be a filename, symbol, config key, literal string, or CLI flag appearing in diff excerpts or commit messages.\n"
        "- Treat mild evaluative wording (e.g., 'simplifies', 'streamlines', 'improves') as acceptable if the diff shows refactoring or consolidation.\n"
        "If evidence shows tests were run/updated (e.g., test files or test commands in diffs/commits/issues), "
        "the description must include a Tests line; if missing, treat as a coverage miss.\n"
        "Correctness and coverage penalties must include a commit SHA or filename evidence anchor.\n"
        "Clarity penalties must include a short evidence quote from the description itself.\n"
        "Clarity issues include only missing required sections or malformed structure (no wording/style critiques).\n"
        "Do NOT include any text outside JSON."
    )
    return system_prompt, user_prompt


def _build_correctness_coverage_prompt(
    commit_payload: List[Dict[str, Any]],
    file_summary_diffs: List[Dict[str, Any]],
    file_list: List[Dict[str, Any]],
    file_list_meta: Dict[str, Any],
    commit_messages: List[Dict[str, Any]],
    file_summaries: List[Dict[str, Any]],
    linked_issues: List[Dict[str, Any]],
    generated: str,
    original: str,
) -> Tuple[str, str]:
    system_prompt = (
        "You are a senior software engineer and strict reviewer. "
        "Judge correctness and coverage only. Use only the provided evidence; do not infer. "
        "Do not use external knowledge or cross-reference anything outside the payload."
    )
    user_prompt = (
        "Commit payload (messages + diff excerpts):\n"
        f"{json.dumps(commit_payload, indent=2)}\n\n"
        "File summary diffs (summaries + diff excerpts):\n"
        f"{json.dumps(file_summary_diffs, indent=2)}\n\n"
        "File list (names + metadata, existence only):\n"
        f"{json.dumps(file_list, indent=2)}\n"
        f"File list meta: {json.dumps(file_list_meta, indent=2)}\n\n"
        "Commit messages:\n"
        f"{json.dumps(commit_messages, indent=2)}\n\n"
        "File summaries:\n"
        f"{json.dumps(file_summaries, indent=2)}\n\n"
        "Linked issues:\n"
        f"{json.dumps(linked_issues, indent=2)}\n\n"
        "Original Description:\n"
        f"{original.strip()}\n\n"
        "Generated Description:\n"
        f"{generated.strip()}\n\n"
        "Use only the evidence above. Do not use external knowledge or cross-reference anything outside this payload.\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "original_correctness_penalty": number,\n'
        '  "generated_correctness_penalty": number,\n'
        '  "original_correctness_penalties": ["unsupported claim ... (evidence: sha/filename)"],\n'
        '  "generated_correctness_penalties": ["unsupported claim ... (evidence: sha/filename)"],\n'
        '  "original_coverage_penalty": number,\n'
        '  "generated_coverage_penalty": number,\n'
        '  "original_coverage_penalties": ["missing key change ... (evidence: sha/filename)"],\n'
        '  "generated_coverage_penalties": ["missing key change ... (evidence: sha/filename)"]\n'
        "}\n"
        "Correctness penalties: 0.5 per unsupported/incorrect claim, cap at 2.0.\n"
        "Coverage penalties: 0.5 per missing key change, cap at 2.0.\n"
        "Return non-negative penalty values (do not use negative numbers).\n"
        "A claim is supported if it is grounded in a diff excerpt OR explicitly stated in a linked issue.\n"
        "File list evidence confirms a file exists/changed but does not prove behavior or content.\n"
        "Do NOT penalize missing documentation-only changes; treat doc/template changes as groupable.\n"
        "High-level summaries are acceptable if they align with the evidence; do not require every minor change.\n"
        "Each Key Changes and Notable Changes bullet must include at least one concrete identifier/token from the evidence.\n"
        "If multiple files/commits changed, require coverage of at least two distinct evidence items.\n"
        "If evidence shows tests were run/updated (e.g., test files or test commands in diffs/commits/issues), "
        "the description must include a Tests line; if missing, treat as a coverage miss.\n"
        "Each penalty must include an evidence anchor referencing a commit SHA or filename from the payload.\n"
        "Do NOT include any text outside JSON."
    )
    return system_prompt, user_prompt


def _build_correctness_coverage_single_prompt(
    commit_payload: List[Dict[str, Any]],
    file_summary_diffs: List[Dict[str, Any]],
    file_list: List[Dict[str, Any]],
    file_list_meta: Dict[str, Any],
    commit_messages: List[Dict[str, Any]],
    file_summaries: List[Dict[str, Any]],
    linked_issues: List[Dict[str, Any]],
    description: str,
) -> Tuple[str, str]:
    system_prompt = (
        "You are a senior software engineer and strict reviewer. "
        "Judge correctness and coverage only. Use only the provided evidence; do not infer. "
        "Do not use external knowledge or cross-reference anything outside the payload."
    )
    user_prompt = (
        "Commit payload (messages + diff excerpts):\n"
        f"{json.dumps(commit_payload, indent=2)}\n\n"
        "File summary diffs (summaries + diff excerpts):\n"
        f"{json.dumps(file_summary_diffs, indent=2)}\n\n"
        "File list (names + metadata, existence only):\n"
        f"{json.dumps(file_list, indent=2)}\n"
        f"File list meta: {json.dumps(file_list_meta, indent=2)}\n\n"
        "Commit messages:\n"
        f"{json.dumps(commit_messages, indent=2)}\n\n"
        "File summaries:\n"
        f"{json.dumps(file_summaries, indent=2)}\n\n"
        "Linked issues:\n"
        f"{json.dumps(linked_issues, indent=2)}\n\n"
        "Description to evaluate:\n"
        f"{description.strip()}\n\n"
        "Use only the evidence above. Do not use external knowledge or cross-reference anything outside this payload.\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "correctness_penalty": number,\n'
        '  "correctness_penalties": ["unsupported claim ... (evidence: sha/filename)"],\n'
        '  "coverage_penalty": number,\n'
        '  "coverage_penalties": ["missing key change ... (evidence: sha/filename)"]\n'
        "}\n"
        "Correctness penalties: 0.5 per unsupported/incorrect claim, cap at 2.0.\n"
        "Coverage penalties: 0.5 per missing key change, cap at 2.0.\n"
        "Return non-negative penalty values (do not use negative numbers).\n"
        "A claim is supported if it is grounded in a diff excerpt OR explicitly stated in a linked issue.\n"
        "File list evidence confirms a file exists/changed but does not prove behavior or content.\n"
        "Do NOT penalize missing documentation-only changes; treat doc/template changes as groupable.\n"
        "High-level summaries are acceptable if they align with the evidence; do not require every minor change.\n"
        "Each Key Changes and Notable Changes bullet must include at least one concrete identifier/token from the evidence.\n"
        "If multiple files/commits changed, require coverage of at least two distinct evidence items.\n"
        "If evidence shows tests were run/updated (e.g., test files or test commands in diffs/commits/issues), "
        "the description must include a Tests line; if missing, treat as a coverage miss.\n"
        "Each penalty must include an evidence anchor referencing a commit SHA or filename from the payload.\n"
        "Do NOT include any text outside JSON."
    )
    return system_prompt, user_prompt


def _build_clarity_single_prompt(
    description: str,
    prior: Dict[str, Any],
) -> Tuple[str, str]:
    system_prompt = (
        "You are a senior software engineer and strict reviewer. "
        "Judge clarity only. Do not consider correctness or coverage. "
        "Use only the provided description; do not use external knowledge."
    )
    user_prompt = (
        "Prior penalties:\n"
        f"{json.dumps(prior, indent=2)}\n\n"
        "Description to evaluate:\n"
        f"{description.strip()}\n\n"
        "Use only the description above. Do not infer facts or use external knowledge.\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "clarity_penalty": number,\n'
        '  "penalties": ["clarity issue ... (evidence: quote)"]\n'
        "}\n"
        "Penalties: 0.5 per clarity issue, cap at 1.0.\n"
        "Return non-negative penalty values (do not use negative numbers).\n"
        "Each penalty must include a short evidence quote from the description itself.\n"
        "Clarity issues include missing required sections or malformed structure.\n"
        "Do NOT include any text outside JSON."
    )
    return system_prompt, user_prompt


def _build_context(
    pr_context: Dict[str, Any],
    record: Dict[str, Any],
    use_cmg_commits: bool = False,
) -> Dict[str, Any]:
    issues = pr_context.get("linked_issues", []) or []
    commits = pr_context.get("commits", []) or []
    files = pr_context.get("files", []) or []

    file_summaries = record.get("file_summaries", []) or []
    rewritten_commits = record.get("rewritten_commits", []) or []

    # Compact context to avoid huge payloads; keep identifiers and summaries.
    ctx: Dict[str, Any] = {
        "linked_issues": [
            {
                "number": i.get("number"),
                "title": i.get("title"),
                "body": i.get("body"),
                "state": i.get("state"),
                "source": i.get("source"),
            }
            for i in issues
        ],
        "rewritten_commits": rewritten_commits,
        "file_summaries": file_summaries,
        "commit_messages": [
            {
                "sha": c.get("sha"),
                "message": (
                    c.get("cmg_rewritten_message")
                    if use_cmg_commits
                    else (c.get("message") or c.get("rewritten_message"))
                ),
            }
            for c in commits
        ],
        "files": [
            {"filename": f.get("filename"), "status": f.get("status")}
            for f in files
        ],
    }
    return ctx


def _build_evidence_payload(
    pr_context: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    commits = pr_context.get("commits", []) or []
    commit_payload = CommitMessageRewriter.build_payload(
        commits,
        use_cmg_commits=False,
        max_lines_per_patch=None,
    )

    files = pr_context.get("files", []) or []
    judge_cfg = judge_defaults.get("evidence", {}) or {}
    max_file_list_items = int(judge_cfg.get("max_file_list_items") or DEFAULT_MAX_FILE_LIST_ITEMS)
    file_list, file_list_truncated = _build_file_list(files, max_items=max_file_list_items)
    file_list_meta = {
        "total_files": len(files),
        "returned_files": len(file_list),
        "truncated": file_list_truncated,
    }

    diff_payload = FileDiffSummarizer.build_payload(files, max_lines_per_patch=None)
    file_summary_diffs: List[Dict[str, Any]] = []
    for diff in diff_payload:
        file_summary_diffs.append(
            {
                "filename": diff.get("filename"),
                "summary": "",
                "status": diff.get("status"),
                "additions": diff.get("additions"),
                "deletions": diff.get("deletions"),
                "diff_excerpt": diff.get("diff_excerpt"),
            }
        )
    return commit_payload, file_summary_diffs, file_list, file_list_meta


def _run_full_judgment_with_budget(
    llm,
    commit_payload: List[Dict[str, Any]],
    file_summary_diffs: List[Dict[str, Any]],
    file_list: List[Dict[str, Any]],
    file_list_meta: Dict[str, Any],
    commit_messages: List[Dict[str, Any]],
    file_summaries: List[Dict[str, Any]],
    linked_issues: List[Dict[str, Any]],
    description: str,
    repo: str,
    pr_number: int,
    max_prompt_tokens: int | None,
    safety_factor: float,
) -> tuple[Dict[str, Any], Any, float, Dict[str, Any]]:
    sys_full, user_full = _build_full_judgment_single_prompt(
        commit_payload,
        file_summary_diffs,
        file_list,
        file_list_meta,
        commit_messages,
        file_summaries,
        linked_issues,
        description,
    )
    estimated = _estimate_prompt_tokens(sys_full, user_full)
    effective_estimate = int(estimated * (safety_factor if safety_factor > 0 else 1.0))
    if not max_prompt_tokens or effective_estimate <= max_prompt_tokens:
        start_full = time.monotonic()
        raw_full = llm.chat(
            system_prompt=sys_full,
            user_prompt=user_full,
            log_type="pr_description_judge_full",
            repo=repo,
            pr_number=pr_number,
        ).strip()
        if raw_full.startswith("[LLM skipped:"):
            return {}, raw_full, 0.0, {
                "chunked": False,
                "estimated_tokens": estimated,
                "effective_estimate": effective_estimate,
                "skipped": True,
                "skip_reason": raw_full,
            }
        dur_full = time.monotonic() - start_full
        parsed_full = _safe_parse_json(raw_full)
        return parsed_full, raw_full, dur_full, {
            "chunked": False,
            "estimated_tokens": estimated,
            "effective_estimate": effective_estimate,
        }

    chunks = _chunk_evidence_by_tokens(
        commit_payload,
        file_summary_diffs,
        commit_messages,
        file_summaries,
        linked_issues,
        description,
        total_files=int(file_list_meta.get("total_files") or 0),
        max_prompt_tokens=max_prompt_tokens,
    )
    raw_chunks: List[str] = []
    parsed_chunks: List[Dict[str, Any]] = []
    durations: List[float] = []

    for idx, chunk in enumerate(chunks, start=1):
        sys_prompt, user_prompt = _build_full_judgment_single_prompt(
            commit_payload,
            chunk.get("file_summary_diffs", []),
            chunk.get("file_list", []),
            chunk.get("file_list_meta", {}),
            commit_messages,
            file_summaries,
            linked_issues,
            description,
        )
        start_full = time.monotonic()
        raw = llm.chat(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            log_type=f"pr_description_judge_full_chunk_{idx}",
            repo=repo,
            pr_number=pr_number,
        ).strip()
        if raw.startswith("[LLM skipped:"):
            return {}, raw, 0.0, {
                "chunked": True,
                "chunk_count": len(chunks),
                "estimated_tokens": estimated,
                "effective_estimate": effective_estimate,
                "skipped": True,
                "skip_reason": raw,
            }
        durations.append(time.monotonic() - start_full)
        raw_chunks.append(raw)
        parsed_chunks.append(_safe_parse_json(raw))

    def _max_penalty(key: str) -> float:
        vals = [float(p.get(key) or 0.0) for p in parsed_chunks if isinstance(p, dict)]
        return max(vals) if vals else 0.0

    correctness_penalty = _max_penalty("correctness_penalty")
    coverage_penalty = _max_penalty("coverage_penalty")
    clarity_penalty = _max_penalty("clarity_penalty")
    correctness_penalties: List[str] = []
    coverage_penalties: List[str] = []
    clarity_penalties: List[str] = []
    for idx, parsed in enumerate(parsed_chunks, start=1):
        for item in (parsed.get("correctness_penalties") or []):
            correctness_penalties.append(f"[chunk {idx}] {item}")
        for item in (parsed.get("coverage_penalties") or []):
            coverage_penalties.append(f"[chunk {idx}] {item}")
        for item in (parsed.get("clarity_penalties") or []):
            clarity_penalties.append(f"[chunk {idx}] {item}")

    aggregated = {
        "correctness_penalty": correctness_penalty,
        "coverage_penalty": coverage_penalty,
        "clarity_penalty": clarity_penalty,
        "correctness_penalties": correctness_penalties,
        "coverage_penalties": coverage_penalties,
        "clarity_penalties": clarity_penalties,
        "chunked": True,
        "chunk_count": len(chunks),
    }
    return aggregated, raw_chunks, float(sum(durations)), {
        "chunked": True,
        "chunk_count": len(chunks),
        "estimated_tokens": estimated,
        "effective_estimate": effective_estimate,
        "chunks": chunks,
        "durations_sec": [round(d, 2) for d in durations],
    }


pipeline_config = load_pipeline_config()
dataset_config = pipeline_config.get("dataset", {}) or {}
dataset_csv = (dataset_config.get("csv_path") or "data/parsed.csv").strip()
dataset_name = (dataset_config.get("name") or Path(dataset_csv).stem).strip()
judge_defaults = pipeline_config.get("judge", {})
llm_defaults = pipeline_config.get("llm", {})
ranking_config = pipeline_config.get("ranking", {}) or {}
commit_payload_config = pipeline_config.get("commit_payload", {}) or {}
default_judge_provider = (
    os.getenv("LLM_PROVIDER")
    or llm_defaults.get("provider")
    or judge_defaults.get("default_provider")
    or "openai"
).lower()

parser = argparse.ArgumentParser(description="Judge generated PR descriptions against originals using an LLM.")
parser.add_argument(
    "--descriptions_path",
    type=Path,
    default=None,
    help="Path to descriptions-*.json. If omitted, uses the latest in results/pr-description/<provider>/",
)
parser.add_argument(
    "--graph_path",
    type=Path,
    default=ROOT_DIR / "results" / "knowledge_graph" / f"graph-{dataset_name}.json",
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--provider",
    type=str,
    default=default_judge_provider,
    choices=["mistral", "openai", "llama", "deepseek", "gemini"],
    help="LLM provider to use for judging (overrides LLM_PROVIDER for this script only).",
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Limit the number of description records processed.",
)
parser.add_argument(
    "--previous",
    action="store_true",
    help="Continue from the latest judge output for the same descriptions file.",
)
args = parser.parse_args()

def main() -> int:
    load_dotenv()

    provider = args.provider.lower()
    results_dir = ROOT_DIR / "results" / "pr-description" / provider
    results_dir.mkdir(parents=True, exist_ok=True)
    llm_log_dir = ROOT_DIR / "logs" / "judge" / "llm"
    judge_evidence_cfg = judge_defaults.get("evidence", {}) or {}
    max_prompt_tokens = int(judge_evidence_cfg.get("max_prompt_tokens") or DEFAULT_MAX_PROMPT_TOKENS)
    safety_factor = float(
        judge_evidence_cfg.get("prompt_token_safety_factor") or DEFAULT_PROMPT_TOKEN_SAFETY_FACTOR
    )
    if max_prompt_tokens <= 0:
        max_prompt_tokens = None

    descriptions_path = args.descriptions_path
    if descriptions_path is None:
        descriptions_path = _find_latest_descriptions_file(results_dir, dataset_name=dataset_name)

    if not descriptions_path.exists():
        raise FileNotFoundError(f"Descriptions file not found: {descriptions_path}")
    print(f"[JUDGE] Using descriptions file: {descriptions_path.absolute()}")

    graph_reader = KnowledgeGraphReader(args.graph_path)

    with open(descriptions_path, "r", encoding="utf-8") as f:
        description_records = json.load(f)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be a positive integer.")
        description_records = description_records[: args.limit]
    total_records = len(description_records) if isinstance(description_records, list) else 0
    unique_prs_in_descriptions = {
        (r.get("repo_name"), r.get("pr_number"))
        for r in (description_records or [])
        if isinstance(r, dict) and r.get("repo_name") and r.get("pr_number") is not None
    }
    print(
        "[JUDGE] Description file contains "
        f"{len(unique_prs_in_descriptions)} PRs across {total_records} records."
    )

    # Initialize LLM client
    judge_temperature = judge_defaults.get("temperature")

    if provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is required for the judge (provider=mistral)")
        llm = MistralClient(api_key, log_dir=str(llm_log_dir), temperature=judge_temperature)
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the judge (provider=openai)")
        llm = OpenAIClient(api_key, log_dir=str(llm_log_dir), temperature=judge_temperature)
    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for the judge (provider=deepseek)")
        llm = DeepSeekClient(api_key, log_dir=str(llm_log_dir), temperature=judge_temperature)
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for the judge (provider=gemini)")
        llm = GeminiClient(api_key, log_dir=str(llm_log_dir), temperature=judge_temperature)
    else:  # llama/local
        llm = LlamaClient(os.getenv("LLM_API_KEY"), log_dir=str(llm_log_dir), temperature=judge_temperature)

    judgments: List[Dict[str, Any]] = []
    processed_prs: set[tuple[str, int]] = set()
    processed_modes: set[tuple[str, int, str]] = set()

    # Prepare output file and write incrementally per PR
    judge_dir = ROOT_DIR / "results" / "judge" / provider
    judge_dir.mkdir(parents=True, exist_ok=True)
    stem = descriptions_path.stem  # e.g., descriptions-1
    previous_file = None
    if args.previous:
        previous_file = _find_latest_judge_file(judge_dir, stem)
        if previous_file and previous_file.exists():
            with previous_file.open("r", encoding="utf-8") as f:
                try:
                    judgments = json.load(f) or []
                except Exception:
                    judgments = []
            for item in judgments:
                repo_name = item.get("repo_name")
                pr_num = item.get("pr_number")
                gen_mode = item.get("generation_mode")
                if repo_name and pr_num is not None and gen_mode:
                    processed_modes.add((repo_name, int(pr_num), str(gen_mode)))
                    processed_prs.add((repo_name, int(pr_num)))
    if previous_file:
        output_file = previous_file
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        output_file = judge_dir / f"{stem}-judge-{timestamp}.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump([], f)

    for record in description_records:
        repo = record.get("repo_name")
        pr_number = record.get("pr_number")
        if not repo or pr_number is None:
            continue

        if processed_modes:
            mode_records = record.get("modes")
            mode_records = mode_records if isinstance(mode_records, list) and mode_records else [record]
            remaining_modes = [
                m
                for m in mode_records
                if (repo, int(pr_number), str(m.get("generation_mode"))) not in processed_modes
            ]
            if not remaining_modes:
                print(f"[JUDGE] Skipping {repo}#{pr_number} (already judged).")
                continue

        print("\n" + "=" * 80)
        print(f"[JUDGE] Repo: {repo} | PR #{pr_number}")
        print("=" * 80 + "\n")

        pr_context = graph_reader.get_pr_context(repo, pr_number)
        original_desc = record.get("original_description", "")

        modes = record.get("modes")
        mode_records = modes if isinstance(modes, list) and modes else [record]

        baseline_record = None
        for candidate in mode_records:
            opts = candidate.get("generation_options", {}) or {}
            if opts.get("use_cmg") and opts.get("include_file_summaries"):
                baseline_record = candidate
                break
        if baseline_record is None:
            for candidate in mode_records:
                opts = candidate.get("generation_options", {}) or {}
                if opts.get("include_file_summaries"):
                    baseline_record = candidate
                    break
        if baseline_record is None:
            baseline_record = mode_records[0]

        baseline_context = _build_context(pr_context, baseline_record, use_cmg_commits=False)
        baseline_original = original_desc
        start_pr = time.monotonic()

        print("[JUDGE][BASELINE] Starting baseline (original description) checks.\n")

        baseline_parsed: Dict[str, Any]
        (
            commit_payload,
            file_summary_diffs,
            file_list,
            file_list_meta,
        ) = _build_evidence_payload(pr_context)
        parsed_full, raw_full, dur_full, chunk_meta = _run_full_judgment_with_budget(
            llm,
            commit_payload,
            file_summary_diffs,
            file_list,
            file_list_meta,
            baseline_context.get("commit_messages", []),
            baseline_context.get("file_summaries", []),
            baseline_context.get("linked_issues", []),
            baseline_original,
            repo,
            int(pr_number),
            max_prompt_tokens,
            safety_factor,
        )
        if chunk_meta.get("skipped"):
            print(f"[JUDGE] Skipping {repo}#{pr_number}: {chunk_meta.get('skip_reason')}")
            continue

        baseline_parsed = {
            "original_score": _score_from_penalties(
                float(parsed_full.get("correctness_penalty") or 0.0),
                float(parsed_full.get("coverage_penalty") or 0.0),
                float(parsed_full.get("clarity_penalty") or 0.0),
            ),
            "original_breakdown": {
                "correctness_penalty": parsed_full.get("correctness_penalty", 0.0),
                "coverage_penalty": parsed_full.get("coverage_penalty", 0.0),
                "clarity_penalty": parsed_full.get("clarity_penalty", 0.0),
                "penalties": (parsed_full.get("correctness_penalties") or [])
                + (parsed_full.get("coverage_penalties") or [])
                + (parsed_full.get("clarity_penalties") or []),
            },
            "steps": {
                "full_raw": raw_full,
                "commit_payload": commit_payload,
                "file_summary_diffs": file_summary_diffs,
                "file_list": file_list,
                "file_list_meta": file_list_meta,
                "file_summaries": [],
                "chunking": chunk_meta,
                "durations_sec": {
                    "full": round(dur_full, 2),
                },
            },
        }

        for mode_record in mode_records:
            generated_desc = mode_record.get("generated_description", "")
            generation_mode = mode_record.get("generation_mode")
            generation_options = mode_record.get("generation_options", {}) or {}
            if (repo, int(pr_number), str(generation_mode)) in processed_modes:
                print(f"[JUDGE][MODE] Skipping already judged mode: {generation_mode}")
                continue

            print("-" * 80)
            print(f"[JUDGE][MODE] {generation_mode}")
            print(f"[JUDGE][MODE] options={generation_options}")
            print("-" * 80 + "\n")

            context_payload = _build_context(pr_context, mode_record, use_cmg_commits=False)

            parsed: Dict[str, Any]
            raw = ""
            (
                commit_payload,
                file_summary_diffs,
                file_list,
                file_list_meta,
            ) = _build_evidence_payload(pr_context)

            parsed_full, raw_full, dur_full, chunk_meta = _run_full_judgment_with_budget(
                llm,
                commit_payload,
                file_summary_diffs,
                file_list,
                file_list_meta,
                context_payload.get("commit_messages", []),
                context_payload.get("file_summaries", []),
                context_payload.get("linked_issues", []),
                generated_desc,
                repo,
                int(pr_number),
                max_prompt_tokens,
                safety_factor,
            )
            if chunk_meta.get("skipped"):
                print(f"[JUDGE] Skipping {repo}#{pr_number}: {chunk_meta.get('skip_reason')}")
                break
            print(f"[JUDGE][MODE {generation_mode}] Duration: {dur_full:.2f}s (full)\n")

            g_score = _score_from_penalties(
                float(parsed_full.get("correctness_penalty") or 0.0),
                float(parsed_full.get("coverage_penalty") or 0.0),
                float(parsed_full.get("clarity_penalty") or 0.0),
            )
            o_score = float(baseline_parsed.get("original_score") or 0.0)
            prefers = "generated" if g_score > o_score else "original"
            parsed = {
                "original_score": o_score,
                "generated_score": g_score,
                "prefers": prefers,
                "reason": "Scores derived from multi-step rubric.",
                "original_breakdown": baseline_parsed.get("original_breakdown"),
                "generated_breakdown": {
                    "correctness_penalty": parsed_full.get("correctness_penalty", 0.0),
                    "coverage_penalty": parsed_full.get("coverage_penalty", 0.0),
                    "clarity_penalty": parsed_full.get("clarity_penalty", 0.0),
                    "penalties": (parsed_full.get("correctness_penalties") or [])
                    + (parsed_full.get("coverage_penalties") or [])
                    + (parsed_full.get("clarity_penalties") or []),
                },
                "steps": {
                    "full_raw": raw_full,
                    "commit_payload": commit_payload,
                    "file_summary_diffs": file_summary_diffs,
                    "file_list": file_list,
                    "file_list_meta": file_list_meta,
                    "file_summaries": [],
                    "chunking": chunk_meta,
                    "durations_sec": {
                        "full": round(dur_full, 2),
                    },
                },
            }

            original_breakdown = parsed.get("original_breakdown") if isinstance(parsed, dict) else None
            generated_breakdown = parsed.get("generated_breakdown") if isinstance(parsed, dict) else None

            judgments.append(
                {
                    "repo_name": repo,
                    "pr_number": pr_number,
                    "generation_mode": generation_mode,
                    "generation_options": generation_options,
                    "original_description": original_desc,
                    "generated_description": generated_desc,
                    "judgment": parsed,
                    "rubric": _rubric_spec(),
                    "analysis": {
                        "original_primary_reason": _primary_reason_from_breakdown(original_breakdown),
                        "generated_primary_reason": _primary_reason_from_breakdown(generated_breakdown),
                    },
                    "raw_response": raw or parsed.get("steps"),
                }
            )
            processed_prs.add((repo, int(pr_number)))
            processed_modes.add((repo, int(pr_number), str(generation_mode)))

            max_seconds = float(judge_defaults.get("max_seconds_per_pr") or DEFAULT_MAX_SECONDS_PER_PR)
            if max_seconds > 0 and (time.monotonic() - start_pr) > max_seconds:
                print(f"[JUDGE] Skipping remaining modes for {repo}#{pr_number} (time limit reached).")
                break

            # Persist after each mode
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(judgments, f, indent=2)

    print(
        f"[JUDGE] Saved judgments to {output_file.absolute()} "
        f"(processed {len(processed_prs)} PRs; "
        f"descriptions file had {len(unique_prs_in_descriptions)} PRs)."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
