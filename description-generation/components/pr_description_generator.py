"""
Single-call generator that summarizes file diffs and produces
the final pull-request description in one LLM invocation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Optional
import re
import difflib

from .commit_message_rewriter import CommitMessageRewriter
from .file_diff_summarizer import FileDiffSummarizer, _is_docs_file
from .patch_utils import combine_patches, clean_patch
from .ranking import compute_file_scores, rank_commits


class PRDescriptionGenerator:
    MAX_LINES_PER_PATCH: int | None = None

    def __init__(self, llm, ranking_config: Dict[str, Any] | None = None) -> None:
        self.llm = llm
        self.ranking_config = ranking_config or {}

    def generate_outputs(
        self,
        pr_context: Dict[str, Any],
        repo_name: str,
        pr_number: int,
        include_file_summaries: bool = True,
        include_commits: bool = True,
        use_cmg_commits: bool = False,
        precomputed_file_summaries: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        print(
            "[PRDescriptionGenerator] Generating PR description "
            f"(commits={'on' if include_commits else 'off'}, "
            f"cmg={'on' if use_cmg_commits else 'off'}, "
            f"file_summaries={'on' if include_file_summaries else 'off'})...\n"
        )

        file_summaries: List[Dict[str, Any]] = []
        condensed_summaries: List[Dict[str, Any]] = []
        if include_file_summaries:
            if precomputed_file_summaries:
                file_summaries = precomputed_file_summaries
                condensed_summaries = self._condense_file_summaries(file_summaries)
            else:
                file_summaries = FileDiffSummarizer.generate_summaries(
                    self.llm,
                    pr_context.get("files") or [],
                    repo_name=repo_name,
                    pr_number=pr_number,
                    max_lines_per_patch=self.MAX_LINES_PER_PATCH,
                )
                condensed_summaries = self._condense_file_summaries(file_summaries)
            for summary in file_summaries:
                if "is_docs" not in summary:
                    summary["is_docs"] = _is_docs_file(summary.get("filename") or "")
        else:
            print("[PRDescriptionGenerator] Skipping file diff summarization (mode excludes file summaries).\n")

        prompt_payload = self._build_payload(
            pr_context,
            include_file_summaries=include_file_summaries,
            include_commits=include_commits,
            use_cmg_commits=use_cmg_commits,
            file_summaries=condensed_summaries or file_summaries,
        )
        system_prompt, user_prompt = self._build_prompts(
            prompt_payload,
            repo_name,
            pr_number,
            include_file_summaries=include_file_summaries,
        )

        raw_response = self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            log_type="pr_single_call_generation",
            repo=repo_name,
            pr_number=pr_number,
        ).strip()

        parsed = self._parse_response(raw_response)

        description = (parsed.get("pr_description") or "").strip()
        rewritten_commits: List[Dict[str, Any]] = []
        if not file_summaries:
            file_summaries = parsed.get("file_summaries") or []

        print("[PRDescriptionGenerator] LLM call complete. Returning structured outputs.\n")
        return {
            "description": description,
            "rewritten_commits": rewritten_commits,
            "file_summaries": file_summaries,
            "raw_response": raw_response,
        }

    # ------------------------------------------------------------------ #
    # Prompt construction helpers
    # ------------------------------------------------------------------ #

    def _build_payload(
        self,
        pr_context: Dict[str, Any],
        include_file_summaries: bool = True,
        include_commits: bool = True,
        use_cmg_commits: bool = False,
        file_summaries: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        linked_issues = pr_context.get("linked_issues", []) or []
        commits = pr_context.get("commits", []) or []
        files = pr_context.get("files", []) or []

        commit_payload: List[Dict[str, Any]] = []
        if include_commits:
            commit_cfg = (self.ranking_config.get("commit") or {})
            include_all_if_leq = int(commit_cfg.get("include_all_if_commit_count_leq") or 0)
            top_k = int(commit_cfg.get("top_k_large") or 0)
            commits_for_prompt = commits
            if top_k and (include_all_if_leq == 0 or len(commits) > include_all_if_leq):
                file_weights = (self.ranking_config.get("file") or {}).get("weights") or {}
                file_scores = compute_file_scores(files, file_weights)
                ranked = rank_commits(commits, file_scores, commit_cfg.get("weights") or {})
                keep_shas = {sha for sha, _ in ranked[:top_k]}
                commits_for_prompt = [c for c in commits if c.get("sha") in keep_shas]
            commit_payload = CommitMessageRewriter.build_payload(
                commits_for_prompt,
                use_cmg_commits=use_cmg_commits,
                max_lines_per_patch=self.MAX_LINES_PER_PATCH,
            )
            commit_max_tokens = int((self.ranking_config.get("commit_payload") or {}).get("max_tokens_per_prompt") or 0)
            commit_payload = CommitMessageRewriter.trim_payload_by_tokens(commit_payload, commit_max_tokens)

        file_payload: List[Dict[str, Any]] = []
        if include_file_summaries and not file_summaries:
            file_payload = FileDiffSummarizer.build_payload(
                files,
                max_lines_per_patch=self.MAX_LINES_PER_PATCH,
            )

        payload: Dict[str, Any] = {
            "linked_issues": [
                {
                    "number": issue.get("number"),
                    "title": issue.get("title"),
                    "state": issue.get("state"),
                    "source": issue.get("source"),
                }
                for issue in linked_issues
            ],
            "files": file_payload,
            "settings": {
                "include_file_summaries": include_file_summaries,
                "include_commits": include_commits,
            },
        }

        if include_file_summaries and file_summaries:
            payload["file_summaries"] = file_summaries
            diff_payload = FileDiffSummarizer.build_payload(
                files,
                max_lines_per_patch=self.MAX_LINES_PER_PATCH,
            )
            diff_map = {item.get("filename"): item for item in diff_payload}
            file_summary_diffs: List[Dict[str, Any]] = []
            for summary in file_summaries:
                filename = summary.get("filename")
                if not filename:
                    continue
                diff = diff_map.get(filename)
                if not diff:
                    continue
                file_summary_diffs.append(
                    {
                        "filename": filename,
                        "summary": summary.get("summary"),
                        "status": diff.get("status"),
                        "additions": diff.get("additions"),
                        "deletions": diff.get("deletions"),
                        "diff_excerpt": diff.get("diff_excerpt"),
                    }
                )
            payload["file_summary_diffs"] = file_summary_diffs

        if include_commits:
            payload["commits"] = commit_payload

        return payload

    @staticmethod
    def _normalize_summary(text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[\"'`]", "", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _condense_file_summaries(
        self,
        summaries: List[Dict[str, Any]],
        max_per_cluster: int = 2,
        similarity_threshold: float = 0.9,
    ) -> List[Dict[str, Any]]:
        clusters: List[Dict[str, Any]] = []
        for item in summaries:
            summary = item.get("summary") or ""
            normalized = self._normalize_summary(summary)
            if not normalized:
                continue
            matched = False
            for cluster in clusters:
                ratio = difflib.SequenceMatcher(a=normalized, b=cluster["normalized"]).ratio()
                if ratio >= similarity_threshold:
                    if len(cluster["items"]) < max_per_cluster:
                        cluster["items"].append(item)
                    matched = True
                    break
            if not matched:
                clusters.append({"normalized": normalized, "items": [item]})

        condensed: List[Dict[str, Any]] = []
        for cluster in clusters:
            condensed.extend(cluster["items"][:max_per_cluster])
        return condensed

    def _build_prompts(
        self,
        payload: Dict[str, Any],
        repo_name: str,
        pr_number: int,
        include_file_summaries: bool = True,
    ) -> Tuple[str, str]:
        system_prompt = (
            "You are a senior software engineer writing a PR description for reviewers. "
            "This is a research pipeline focused on grounded, audit-friendly summaries.\n"
            "You must produce machine-parseable JSON only.\n\n"
            "Purpose:\n"
            "- Turn the provided PR context into a concise, reviewer-ready description.\n"
            "- Prioritize correctness, coverage of key changes, and clarity for quick review.\n\n"
            "Grounding rules (absolute):\n"
            "- Use only facts present in the provided context. Do not infer or invent.\n"
            "- If something is missing, omit it rather than guessing.\n"
            "- Do not use any external knowledge or cross-reference anything outside the provided context.\n"
            "- Do not reference PR body text (it is intentionally excluded).\n\n"
            "Output rules (absolute):\n"
            "- Respond with a single JSON object only.\n"
            "- Do NOT include any explanation, prose, or code outside the JSON object.\n"
            "- Do NOT wrap the JSON in Markdown code fences.\n"
            "- Escape any newline characters inside string values as \\n so that the JSON is valid."
        )

        schema_description = json.dumps(
            {
                "file_summaries": [
                    {"filename": "string", "summary": "≤18 word sentence"}
                ],
                "pr_description": "markdown string (≤120 words, structured as instructed)",
                "evidence_anchors": [
                    {"item": "short claim text", "evidence": "sha or filename"}
                ],
            },
            indent=2,
        )

        payload_text = json.dumps(payload, indent=2)

        if include_file_summaries:
            file_rules = (
                "- For file summaries: they are already provided in the context. Return an empty list [].\n"
                "- Only mention files that appear in `file_summary_diffs`.\n"
                "- When describing file changes, rely on the `diff_excerpt` fields.\n"
            )
        else:
            file_rules = (
                "- For file summaries: file diffs are omitted for this mode. Return an empty list [].\n"
                "- Do not reference any specific files.\n"
            )

        user_prompt = (
            f"Repository: {repo_name}\n"
            f"Pull Request Number: {pr_number}\n\n"
            "Context (authoritative, do not go beyond this):\n"
            f"{payload_text}\n\n"
            "Output Requirements:\n"
            "Return a single JSON object with exactly these keys:\n"
            "- `file_summaries`: list of {filename, summary}\n"
            "- `pr_description`: markdown string\n\n"
            "- `evidence_anchors`: list of {item, evidence}\n\n"
            "Schema (for reference):\n"
            f"{schema_description}\n\n"
            "Rules:\n"
            f"{file_rules}"
            "- PR description format: Markdown with sections `### Summary`, `### Key Changes`, `### Notable Changes`, `### Linked Issues`.\n"
            "- Summary: 1–2 sentences, plain language, no speculation.\n"
            "- Summary must include at least one concrete identifier/token from diff excerpts or file summaries.\n"
            "- Add a `Tests:` line under Summary only when explicit test names/commands are evidenced (e.g., `pytest`, `mvn test`, `go test`, specific test files).\n"
            "- If the context only says testing was done (e.g., 'tested', 'personally tested') without specifics, omit the Tests line.\n"
            "- Key Changes: up to 3 bullets, each ≤15 words.\n"
            "- Notable Changes: up to 2 bullets, each ≤15 words.\n"
            "- Each Key Changes and Notable Changes bullet must include at least one concrete identifier/token from the diff excerpts or file summaries.\n"
            "- Notable Changes must reference concrete identifiers or tokens that appear in the diff excerpts (use exact names/terms from the diffs).\n"
            "- Do NOT add Notable Changes that are not explicitly evidenced in the context.\n"
            "- Linked Issues: issue numbers if present, otherwise 'None'.\n"
            "- Entire description target length 120–160 words (hard cap 170 words).\n"
            "- Only use commit information if it appears in the `commits` section of the context.\n"
            "- If `linked_issues` is empty, list 'None' and do not imply motivation.\n"
            "- If the context explicitly states a purpose/goal, include it in Summary; otherwise omit it.\n"
            "- Only use linked_issues for motivation/intent; do not infer intent from diffs unless explicitly stated.\n"
            "- If the context explicitly states a reason/why, include it in Summary; otherwise omit it.\n"
            "- If tests are explicitly mentioned in commits, issues, or diffs with concrete names/commands, include a brief Tests note; otherwise omit it.\n"
            "- If there are linked issues, reference only those issue numbers; do not invent links.\n"
            "- Do NOT mention branch names, base branches, repo language, or labels unless they appear explicitly in the context fields.\n"
            "- Do NOT include evidence anchors inside `pr_description`.\n"
            "- Evidence anchors must be returned in `evidence_anchors` only.\n"
            "- For each Key Changes bullet, add one evidence_anchors entry with the bullet text and its sha/filename.\n"
            "- For each Notable Changes bullet, add one evidence_anchors entry with the bullet text and its sha/filename.\n"
            "- If a Tests line is present, add one evidence_anchors entry with the Tests text and its sha/filename.\n"
            "- Respond with valid JSON only. Do NOT include any explanatory text before or after the JSON.\n"
            "- Do NOT wrap the JSON in ``` code fences.\n"
            "- Escape newline characters inside string values as \\n so that the JSON parses."
        )

        return system_prompt, user_prompt

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """
        Robustly parse the LLM response into a JSON object.

        Strategy:
        1. Try direct json.loads(raw).
        2. Try extracting a fenced or braced JSON block.
        3. For each candidate, also try a 'fixed' version where newlines
           inside quoted strings are escaped as '\\n'.
        4. If everything fails, fall back to treating the whole response
           as the PR description text.
        """
        # 1) Try raw as-is
        parsed = self._try_parse_json(raw)
        if parsed is not None:
            return parsed

        # 2) Try to extract a code-fenced JSON block (```json ... ``` or ``` ... ```)
        fenced = self._extract_fenced_block(raw)
        if fenced is not None:
            parsed = self._try_parse_json(fenced)
            if parsed is not None:
                return parsed

        # 3) Try to extract the first {...} block
        braced = self._extract_braced_block(raw)
        if braced is not None:
            parsed = self._try_parse_json(braced)
            if parsed is not None:
                return parsed

        print("[PRDescriptionGenerator] WARNING: Failed to parse LLM JSON response. Returning empty defaults.")
        return {
            "commit_messages": [],
            "file_summaries": [],
            "pr_description": raw.strip(),
        }

    def _try_parse_json(self, candidate: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse a string as JSON, with an additional pass that
        escapes newlines inside quoted strings (common LLM mistake).
        """
        candidate = candidate.strip()

        # First attempt: as-is
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Second attempt: escape newlines inside quoted strings
        fixed = self._escape_newlines_in_strings(candidate)
        if fixed != candidate:
            try:
                obj = json.loads(fixed)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        return None

    def _extract_fenced_block(self, raw: str) -> Optional[str]:
        """
        Extract the first ```json ... ``` or ``` ... ``` block if present.
        """
        # Prefer ```json fences
        m = re.search(r"```json\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # Fall back to any ``` fenced block
        m = re.search(r"```(.*?)```", raw, flags=re.DOTALL)
        if m:
            return m.group(1).strip()

        return None

    def _extract_braced_block(self, raw: str) -> Optional[str]:
        """
        Extract the first {...} block (greedy) as a last-resort candidate.
        """
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            return m.group(0).strip()
        return None

    def _escape_newlines_in_strings(self, s: str) -> str:
        """
        Walk the string and, when inside a double-quoted JSON string,
        replace literal newlines with '\\n'. This makes many 'almost JSON'
        LLM outputs parseable without changing structure.
        """
        result: List[str] = []
        in_string = False
        escape = False

        for ch in s:
            if in_string:
                if escape:
                    # Whatever follows a backslash is taken as-is
                    result.append(ch)
                    escape = False
                else:
                    if ch == "\\":
                        result.append(ch)
                        escape = True
                    elif ch == '"':
                        result.append(ch)
                        in_string = False
                    elif ch == "\n":
                        # Convert newline inside string literal to \n
                        result.append("\\n")
                    elif ch == "\r":
                        # Drop bare \r inside strings or turn into \n
                        result.append("\\n")
                    else:
                        result.append(ch)
            else:
                result.append(ch)
                if ch == '"':
                    in_string = True
                    escape = False

        return "".join(result)

    # ------------------------------------------------------------------ #
    # Diff utilities
    # ------------------------------------------------------------------ #

    def _clean_patch(self, patch: str) -> str:
        return clean_patch(patch, self.MAX_LINES_PER_PATCH)

    def _combine_patches(self, patches: List[Dict[str, Any]]) -> str:
        return combine_patches(patches, self.MAX_LINES_PER_PATCH)
