"""
Builds file payloads for the LLM to summarize file diffs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import re
from pathlib import Path
from .patch_utils import clean_patch
from .ranking import (
    file_score,
    is_ci_path,
    is_noise_path,
    is_test_path,
    path_has_keyword,
    select_top_files,
)


DOC_EXTENSIONS = {".md", ".rst", ".adoc", ".txt"}
SMALL_PR_FILE_LIMIT = 15
TOP_N_FILES_LARGE_PR = 25
LANG_BY_EXT = {
    ".py": "python",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rb": "ruby",
    ".cs": "csharp",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".rs": "rust",
    ".scala": "scala",
}
PUBLIC_PATTERNS = {
    "java": re.compile(r"\bpublic\s+(class|interface|enum|record|void|static|final)\b"),
    "kotlin": re.compile(r"\bpublic\s+(class|interface|object|fun|val|var)\b"),
    "javascript": re.compile(r"\bexport\s+(default|class|function|const|let|var)\b"),
    "typescript": re.compile(r"\bexport\s+(default|class|function|const|let|var|interface|type)\b"),
    "python": re.compile(r"^\s*(def|class)\s+[A-Za-z_][A-Za-z0-9_]*\s*\("),
    "go": re.compile(r"^\s*(func|type)\s+[A-Z][A-Za-z0-9_]*\b"),
    "ruby": re.compile(r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\b"),
    "csharp": re.compile(r"\bpublic\s+(class|interface|struct|enum|void)\b"),
    "cpp": re.compile(r"\b(class|struct|enum)\s+[A-Za-z_][A-Za-z0-9_]*\b"),
    "rust": re.compile(r"\bpub\s+(struct|enum|fn|trait|mod)\b"),
    "scala": re.compile(r"\b(public\s+)?(class|trait|object|def)\b"),
}


def _is_docs_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in DOC_EXTENSIONS


def _language_for_file(path: str) -> str | None:
    ext = Path(path).suffix.lower()
    return LANG_BY_EXT.get(ext)


def _count_public_symbols(patch: str, language: str | None) -> tuple[int, int]:
    if not patch or not language:
        return 0, 0
    pattern = PUBLIC_PATTERNS.get(language)
    if not pattern:
        return 0, 0
    added = 0
    removed = 0
    for line in patch.splitlines():
        if line.startswith(("diff --git", "@@", "+++", "---")):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            if pattern.search(line[1:]):
                added += 1
        elif line.startswith("-") and not line.startswith("---"):
            if pattern.search(line[1:]):
                removed += 1
    return added, removed


class FileDiffSummarizer:
    BATCH_SIZE_SMALL = 2
    BATCH_SIZE_LARGE = 3
    DOCS_TOP_K = 5
    MAX_DIFF_LINES_PER_PROMPT: int | None = None
    RANKING_CONFIG: Dict[str, Any] = {}

    @staticmethod
    def generate_summaries(
        llm,
        files: List[Dict[str, Any]],
        repo_name: str,
        pr_number: int,
        max_lines_per_patch: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        payload = FileDiffSummarizer.build_payload(files, max_lines_per_patch=max_lines_per_patch)
        summaries: List[Dict[str, Any]] = []
        if len(payload) <= SMALL_PR_FILE_LIMIT:
            batch_size = max(1, int(FileDiffSummarizer.BATCH_SIZE_SMALL))
        else:
            batch_size = max(1, int(FileDiffSummarizer.BATCH_SIZE_LARGE))

        max_diff_lines = FileDiffSummarizer.MAX_DIFF_LINES_PER_PROMPT or 0

        def _chunk_by_diff_lines(items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
            if max_diff_lines <= 0:
                return [items[idx: idx + batch_size] for idx in range(0, len(items), batch_size)]
            chunks: List[List[Dict[str, Any]]] = []
            current: List[Dict[str, Any]] = []
            current_lines = 0
            for item in items:
                diff_lines = len((item.get("diff_excerpt") or "").splitlines())
                if not current:
                    current = [item]
                    current_lines = diff_lines
                    if current_lines >= max_diff_lines:
                        chunks.append(current)
                        current = []
                        current_lines = 0
                    continue
                if current_lines + diff_lines > max_diff_lines:
                    chunks.append(current)
                    current = [item]
                    current_lines = diff_lines
                    if current_lines >= max_diff_lines:
                        chunks.append(current)
                        current = []
                        current_lines = 0
                    continue
                current.append(item)
                current_lines += diff_lines
            if current:
                chunks.append(current)
            return chunks

        def _parse_single(raw: str) -> Dict[str, Any] | None:
            raw = (raw or "").strip()
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(raw[start:end + 1])
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            return None

        def _single_summary(item: Dict[str, Any]) -> Dict[str, Any] | None:
            filename = item.get("filename") or ""
            diff_excerpt = item.get("diff_excerpt") or ""
            if not filename or not diff_excerpt:
                return None
            system_prompt = (
                "You are a senior software engineer. Summarize a single file diff into one short sentence. "
                "Use only facts visible in the diff. Do not use external knowledge or cross-reference anything outside the diff."
            )
            user_prompt = (
                f"Repository: {repo_name}\n"
                f"Pull Request Number: {pr_number}\n"
                f"Filename: {filename}\n"
                f"Status: {item.get('status')}\n"
                f"Additions: {item.get('additions')}\n"
                f"Deletions: {item.get('deletions')}\n\n"
                "Diff excerpt:\n"
                f"{diff_excerpt}\n\n"
                "Rules:\n"
                "- Return JSON only with keys: summary.\n"
                "- `summary` must be a single sentence (<=18 words).\n"
                "- Use only information visible in the diff excerpt.\n"
                "- Do not use external knowledge or cross-reference anything outside the diff.\n"
                "- If the file is documentation, summarize the textual change in plain terms.\n"
                "- Do not infer intent unless the diff explicitly states it.\n"
                "- Return only JSON, no extra text.\n"
            )
            summary = llm.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                log_type="file_diff_summary",
                repo=repo_name,
                pr_number=pr_number,
            ).strip()
            parsed = _parse_single(summary)
            if not parsed:
                return {"filename": filename, "summary": summary}
            return {
                "filename": filename,
                "summary": parsed.get("summary") or "",
            }

        def _parse_batch(raw: str) -> List[Dict[str, Any]]:
            raw = (raw or "").strip()
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(raw[start:end + 1])
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
            return []

        def _summarize_batch(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            blocks = []
            for idx, item in enumerate(items, start=1):
                blocks.append(
                    "File {idx}:\n"
                    "Filename: {filename}\n"
                    "Status: {status}\n"
                    "Additions: {additions}\n"
                    "Deletions: {deletions}\n"
                    "Diff excerpt:\n{diff}\n".format(
                        idx=idx,
                        filename=item.get("filename"),
                        status=item.get("status"),
                        additions=item.get("additions"),
                        deletions=item.get("deletions"),
                        diff=item.get("diff_excerpt"),
                    )
                )
            payload = "\n---\n".join(blocks)
            system_prompt = (
                "You are a senior software engineer. Summarize each file diff into one short sentence. "
                "Use only facts visible in the diffs. Do not use external knowledge or cross-reference anything outside the diffs."
            )
            user_prompt = (
                f"Repository: {repo_name}\n"
                f"Pull Request Number: {pr_number}\n\n"
                "Files:\n"
                f"{payload}\n\n"
                "Rules:\n"
                "- Return JSON array with objects: {filename, summary}.\n"
                "- Return exactly one summary per input file, in the same order.\n"
                "- Treat each file independently. Do NOT mix changes between files.\n"
                "- Use only the content under that file's section when writing its summary.\n"
                "- Do NOT reference other files in a given file's summary.\n"
                "- Each summary must be a single sentence (<=18 words).\n"
                "- Use only information visible in each file's diff excerpt.\n"
                "- Do not use external knowledge or cross-reference anything outside the diffs.\n"
                "- If a file is documentation, summarize the textual change in plain terms.\n"
                "- Do not infer intent unless the diff explicitly states it.\n"
                "- Return only JSON, no extra text.\n"
            )
            raw = llm.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                log_type="file_diff_summary_batch",
                repo=repo_name,
                pr_number=pr_number,
            ).strip()
            parsed = _parse_batch(raw)
            if parsed:
                return parsed
            # Fallback to single-file calls if the batch response is malformed.
            fallback: List[Dict[str, Any]] = []
            for item in items:
                single = _single_summary(item)
                if single:
                    fallback.append(single)
            return fallback

        for chunk in _chunk_by_diff_lines(payload):
            if len(chunk) == 1:
                single = _single_summary(chunk[0])
                if single:
                    summaries.append(single)
                continue
            summaries.extend(_summarize_batch(chunk))
        is_docs_by_name = {item.get("filename"): item.get("is_docs") for item in payload}
        for summary in summaries:
            filename = summary.get("filename")
            if filename in is_docs_by_name:
                summary["is_docs"] = bool(is_docs_by_name.get(filename))
        return summaries

    @staticmethod
    def build_payload(
        files: List[Dict[str, Any]],
        max_lines_per_patch: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        scored_files: List[Dict[str, Any]] = []
        scored_docs: List[Dict[str, Any]] = []

        for file in files:
            filename = file.get("filename") or ""
            if not filename:
                continue
            if is_noise_path(filename):
                continue
            patch = (file.get("patch") or "").strip()
            if not patch or patch.lower() == "[patch not available]":
                continue

            cleaned_patch = clean_patch(patch, max_lines_per_patch)
            if not cleaned_patch:
                continue
            language = _language_for_file(filename)
            public_added, public_removed = _count_public_symbols(patch, language)
            is_docs = _is_docs_file(filename)
            is_api_impact = public_added > 0 or public_removed > 0
            is_test = is_test_path(filename)
            is_ci = is_ci_path(filename)
            path_risky = path_has_keyword(filename)
            is_noise = is_noise_path(filename)
            is_important = is_api_impact or is_test or is_ci or path_risky
            weights = (
                (FileDiffSummarizer.RANKING_CONFIG.get("file") or {}).get("weights")
                or {}
            )
            score = file_score(
                {
                    "changes": file.get("changes"),
                    "status": file.get("status"),
                    "is_api_impact": is_api_impact,
                    "path_keyword": path_risky,
                    "is_test": is_test,
                    "is_ci": is_ci,
                    "is_noise": is_noise,
                },
                weights,
            )
            diff_lines = len(cleaned_patch.splitlines())
            entry = {
                "filename": filename,
                "status": file.get("status"),
                "additions": file.get("additions"),
                "deletions": file.get("deletions"),
                "diff_excerpt": cleaned_patch,
                "is_docs": is_docs,
                "is_important": is_important,
                "diff_lines": diff_lines,
                "score": score,
            }
            if is_docs:
                scored_docs.append(entry)
                continue

            scored_files.append(
                {
                    **entry,
                    "is_docs": False,
                }
            )

        ranking_cfg = FileDiffSummarizer.RANKING_CONFIG.get("file") or {}
        include_all_if_leq = int(ranking_cfg.get("include_all_if_file_count_leq") or SMALL_PR_FILE_LIMIT)
        if len(scored_files) <= include_all_if_leq:
            for f in scored_files:
                payload.append(
                    {
                        "filename": f["filename"],
                        "status": f["status"],
                        "additions": f["additions"],
                        "deletions": f["deletions"],
                        "diff_excerpt": f["diff_excerpt"],
                        "is_docs": False,
                    }
                )
        else:
            top_k = int(ranking_cfg.get("top_k_large") or TOP_N_FILES_LARGE_PR)
            always_include = [f["filename"] for f in scored_files if f.get("is_important")]
            picked = select_top_files(scored_files, top_k, always_include)

            for f in picked:
                payload.append(
                    {
                        "filename": f["filename"],
                        "status": f["status"],
                        "additions": f["additions"],
                        "deletions": f["deletions"],
                        "diff_excerpt": f["diff_excerpt"],
                        "is_docs": False,
                    }
                )

        docs_top_k = max(0, int(FileDiffSummarizer.DOCS_TOP_K))
        if docs_top_k and scored_docs:
            scored_docs.sort(key=lambda f: (f.get("diff_lines") or 0, f.get("score") or 0.0), reverse=True)
            for f in scored_docs[:docs_top_k]:
                payload.append(
                    {
                        "filename": f["filename"],
                        "status": f["status"],
                        "additions": f["additions"],
                        "deletions": f["deletions"],
                        "diff_excerpt": f["diff_excerpt"],
                        "is_docs": True,
                    }
                )

        return payload
