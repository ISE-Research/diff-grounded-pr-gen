from __future__ import annotations

import json
import re
import os
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

def _log(msg: str) -> None:
    print(msg)
    print()

from networkx.readwrite import json_graph

from components.cmg_quality import (
    needs_cmg,
    is_merge_or_revert,
    CmgQuality,
    is_good_commit_message,
)


def _postprocess_line(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"(?i)^here\s+is[^:]*:\s*", "", s)
    s = re.sub(r'^[#>*\-\s]+', "", s)
    s = s.replace("#", ".")
    s = s.splitlines()[0] if "\n" in s else s
    return s.strip().rstrip(".")


def _build_diff(patches) -> str:
    lines = []
    for p in patches or []:
        filename = p.get("filename") or ""
        if filename:
            lines.append(f"<FILE> {filename}")
        patch = p.get("patch") or ""
        for line in patch.splitlines():
            if line.startswith(("diff --git", "@@", "+++", "---")):
                continue
            if line.startswith(" "):
                # Keep limited context lines for better grounding.
                lines.append("<CTX> " + line[1:])
            elif line.startswith("+"):
                lines.append("<ADD> " + line[1:])
            elif line.startswith("-"):
                lines.append("<DEL> " + line[1:])
    return "\n".join(lines)[:40000]


_WS_SPLIT = re.compile(r"[^\w./#:+-]+")


def _tokens(txt: str) -> List[str]:
    if not txt:
        return []
    return [t.lower() for t in _WS_SPLIT.split(txt) if t]

def _expand_dotted(tokens: List[str]) -> List[str]:
    expanded: List[str] = []
    for token in tokens:
        if "." in token:
            parts = token.split(".")
        elif "#" in token:
            parts = token.split("#")
        elif "::" in token:
            parts = token.split("::")
        else:
            parts = [token]
        expanded.extend([t for t in parts if t])
    return expanded


def _cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    denom = (float((a**2).sum()) ** 0.5) * (float((b**2).sum()) ** 0.5)
    if denom <= 0:
        return 0.0
    return float((a * b).sum()) / denom


class _GraphDemoPool:
    """
    Loads commit-level demos from the knowledge graph and selects top-k by token overlap.
    """

    def __init__(self, graph_path: str, log_enabled: bool = False):
        self.name = "graph"
        self.log_enabled = log_enabled
        self.recs: List[Dict[str, Any]] = []
        self._tokens: List[set[str]] = []
        self._token_lists: List[List[str]] = []
        self._bm25_idf: Dict[str, float] = {}
        self._bm25_avgdl: float = 1.0
        self._bm25_k1: float = 1.5
        self._bm25_b: float = 0.75
        self._embeddings = None
        self._st = None
        self._np = None

        path = Path(graph_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge graph not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        graph = json_graph.node_link_graph(data, edges="links")

        commit_to_pr: Dict[str, Tuple[str | None, int | None]] = {}
        for pr_id, commit_id, edge_data in graph.edges(data=True):
            if edge_data.get("label") != "PR_CONTAINS_COMMIT":
                continue
            pr_node = graph.nodes.get(pr_id, {})
            repo_name = pr_node.get("repo")
            pr_number = pr_node.get("number")
            commit_to_pr[commit_id] = (repo_name, pr_number)

        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("label") != "Commit":
                continue
            msg = (attrs.get("message") or "").strip()
            is_good = attrs.get("cmg_is_good")
            if is_good is None:
                is_good = is_good_commit_message(msg)
            if not is_good:
                continue
            diff = _build_diff(attrs.get("patches") or [])
            if not diff:
                continue
            repo_name, pr_number = commit_to_pr.get(node_id, (None, None))
            diff_token_count = attrs.get("cmg_diff_token_count")
            if diff_token_count is None:
                diff_token_count = len(_tokens(diff))
            files_touched = attrs.get("cmg_files_touched") or len(attrs.get("files_touched") or [])
            self.recs.append(
                {
                    "diff": diff,
                    "message": msg,
                    "repo": repo_name,
                    "pr_number": pr_number,
                    "diff_token_count": diff_token_count,
                    "files_touched": files_touched,
                }
            )
            toks = _tokens(diff)
            self._token_lists.append(toks)
            self._tokens.append(set(toks))

        self.corpus_size = len(self.recs)
        if self.corpus_size:
            self._build_bm25()
        if self.log_enabled:
            _log(f"[CMG][RET] graph loaded | corpus_size={self.corpus_size} | graph={path}")

    def _build_bm25(self) -> None:
        df: Dict[str, int] = {}
        total_len = 0
        for toks in self._token_lists:
            total_len += len(toks)
            seen = set(toks)
            for t in seen:
                df[t] = df.get(t, 0) + 1
        self._bm25_avgdl = max(1.0, total_len / max(1, len(self._token_lists)))
        idf: Dict[str, float] = {}
        n_docs = max(1, len(self._token_lists))
        for t, doc_freq in df.items():
            idf[t] = math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        self._bm25_idf = idf

    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        freqs: Dict[str, int] = {}
        for t in doc_tokens:
            freqs[t] = freqs.get(t, 0) + 1
        score = 0.0
        doc_len = len(doc_tokens)
        denom_base = self._bm25_k1 * (1 - self._bm25_b + self._bm25_b * (doc_len / self._bm25_avgdl))
        for t in query_tokens:
            if t not in freqs:
                continue
            idf = self._bm25_idf.get(t, 0.0)
            tf = freqs[t]
            score += idf * ((tf * (self._bm25_k1 + 1)) / (tf + denom_base))
        return score

    def _ensure_embeddings(self) -> None:
        if self._embeddings is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self._st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._np = np
            self._embeddings = self._st.encode([r["diff"] for r in self.recs], normalize_embeddings=False)
        except Exception:
            self._st = None
            self._np = None
            self._embeddings = None

    def top_k(
        self,
        query_diff: str,
        k: int = 16,
        scope: str = "global",
        repo_name: str | None = None,
        pr_number: int | None = None,
        files_touched: int | None = None,
    ) -> List[Tuple[str, str]]:
        if not self.recs:
            return []
        q_tokens = set(_tokens(query_diff))
        q_diff_tokens = len(q_tokens)
        q_token_list = _tokens(query_diff)
        scored = []
        bm25_scores: List[Tuple[int, float]] = []
        sem_scores: Dict[int, float] = {}
        self._ensure_embeddings()
        q_emb = None
        if self._embeddings is not None and self._st is not None and self._np is not None:
            q_emb = self._st.encode([query_diff], normalize_embeddings=False)[0]
        for idx, tok in enumerate(self._tokens):
            rec = self.recs[idx]
            if repo_name and pr_number is not None:
                if rec.get("repo") == repo_name and rec.get("pr_number") == pr_number:
                    continue
            if scope == "pr":
                if rec.get("repo") != repo_name or rec.get("pr_number") != pr_number:
                    continue
            elif scope == "repo":
                if rec.get("repo") != repo_name:
                    continue

            if not tok or not q_tokens:
                jaccard = 0.0
            else:
                inter = len(tok & q_tokens)
                union = len(tok | q_tokens)
                jaccard = (inter / union) if union else 0.0

            bm25 = self._bm25_score(q_token_list, self._token_lists[idx])
            bm25_scores.append((idx, bm25))
            if q_emb is not None and self._embeddings is not None:
                sem_scores[idx] = _cosine(self._np.array(q_emb), self._np.array(self._embeddings[idx]))

            # Bonus for similar change size (token count + files touched).
            demo_tokens = rec.get("diff_token_count") or 0
            denom = max(q_diff_tokens, demo_tokens, 1)
            size_bonus = 1.0 - min(abs(q_diff_tokens - demo_tokens) / denom, 1.0)

            demo_files = rec.get("files_touched") or 0
            files_bonus = 0.0
            if files_touched is not None:
                files_den = max(files_touched, demo_files, 1)
                files_bonus = 1.0 - min(abs(files_touched - demo_files) / files_den, 1.0)

            score = jaccard + (0.2 * size_bonus) + (0.1 * files_bonus)
            scored.append((score, idx))

        if bm25_scores:
            bm25_vals = [s for _, s in bm25_scores]
            bm25_min = min(bm25_vals)
            bm25_max = max(bm25_vals)
        else:
            bm25_min = bm25_max = 0.0

        reranked = []
        bm25_map = dict(bm25_scores)
        for base_score, idx in scored:
            bm25_raw = bm25_map.get(idx, 0.0)
            if bm25_max > bm25_min:
                bm25_norm = (bm25_raw - bm25_min) / (bm25_max - bm25_min)
            else:
                bm25_norm = 0.0
            sem_norm = sem_scores.get(idx, 0.0)
            combined = (0.5 * base_score) + (0.25 * bm25_norm) + (0.25 * sem_norm)
            reranked.append((combined, idx))

        reranked.sort(reverse=True)
        chosen = [self.recs[i] for _, i in reranked[:k]]
        return [(r["diff"], r["message"]) for r in chosen]


class CommitMessageRewriter:
    """
    ICL commit message generator using graph demos with a quality gate:
      - Heuristic/semantic score
      - Optional LLM-as-judge fallback (binary or pairwise)
    """

    def __init__(
        self,
        llm_client=None,
        settings: Dict[str, Any] | None = None,
    ):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.settings = settings or {}
        if llm_client is None:
            raise RuntimeError("ICL backend requires llm_client")

        self.log_enabled = bool(self.settings.get("log", False))
        self.k = int(self.settings.get("k", 16))
        self.max_k = int(self.settings.get("max_k", 3))
        self.batch_enabled = bool(self.settings.get("batch_enabled", True))
        self.max_chunk_tokens = int(self.settings.get("max_chunk_tokens", 16000))
        self.batch_demo_k = self.settings.get("batch_demo_k", 2)
        self.max_commits_per_chunk = self.settings.get("max_commits_per_chunk")
        self.debug_demos = max(0, int(self.settings.get("debug_demos", 3)))
        root_dir = Path(__file__).resolve().parents[2]
        graph_path = self.settings.get("graph_path", "results/knowledge_graph/graph.json")
        self.graph_path = str(root_dir / graph_path) if not Path(graph_path).is_absolute() else graph_path
        self.demo_scope = (self.settings.get("demo_scope") or "global").lower()
        self.demo_pool = _GraphDemoPool(self.graph_path, log_enabled=self.log_enabled)
        self.llm = llm_client
        self.qa = CmgQuality(
            llm_client=llm_client,
            settings=self.settings.get("qa"),
            log_enabled=self.log_enabled,
            sem_model_name=self.settings.get("sem_model"),
        )

        retriever_name = getattr(self.demo_pool, "name", "?")
        corpus_size = getattr(self.demo_pool, "corpus_size", "?")
        if self.log_enabled:
            _log(
                "[CMG][INIT] backend=icl-llm4cmg "
                f"retriever={retriever_name} corpus_size={corpus_size} k={self.k} max_k={self.max_k} "
                f"graph_path={self.graph_path}"
            )

    def _estimate_tokens(self, text: str) -> int:
        # Heuristic: ~4 chars per token.
        return max(1, int(len(text) / 4))

    def _effective_k(self) -> int:
        corpus_size = int(getattr(self.demo_pool, "corpus_size", 0) or 0)
        if corpus_size <= 0:
            return min(self.k, self.max_k)
        return min(self.k, self.max_k, corpus_size)

    def _grounded_in_diff(self, message: str, diff_text: str) -> bool:
        """
        Basic grounding check: require at least one content token in the message
        to appear in the diff tokens.
        """
        stop = {
            "add", "adds", "added", "fix", "fixes", "fixed", "update", "updates", "updated",
            "refactor", "remove", "removed", "rename", "renamed", "revert", "improve",
            "optimize", "document", "bump", "implement", "enable", "disable", "handle",
            "support", "streamline", "migrate", "correct", "clean", "extract",
            "change", "changes", "adjust", "tweak", "misc",
        }
        msg_tokens = [t for t in _tokens(message) if t and t not in stop and len(t) > 2]
        if not msg_tokens:
            return True
        diff_tokens = set(_expand_dotted(_tokens(diff_text)))
        msg_tokens = _expand_dotted(msg_tokens)
        return any(t in diff_tokens for t in msg_tokens)

    def _strict_grounding_ok(self, message: str, diff_text: str, min_overlap_tokens: int = 1) -> bool:
        """
        Soft grounding check: require at least one non-trivial content token to appear in the diff.
        """
        if not message:
            return False
        stop = {
            "add", "adds", "added", "fix", "fixes", "fixed", "update", "updates", "updated",
            "refactor", "remove", "removed", "rename", "renamed", "revert", "improve",
            "optimize", "document", "bump", "implement", "enable", "disable", "handle",
            "support", "streamline", "migrate", "correct", "clean", "extract",
            "change", "changes", "adjust", "tweak", "misc",
        }
        diff_tokens = set(_expand_dotted(_tokens(diff_text)))
        msg_tokens = [t for t in _tokens(message) if t and len(t) > 2]
        if not msg_tokens:
            return False
        content = [t for t in _expand_dotted(msg_tokens) if t not in stop]
        if not content:
            return True
        overlap = [t for t in content if t in diff_tokens]
        return len(overlap) >= min_overlap_tokens

    def _format_prompt(self, diff_text: str, demos: List[Tuple[str, str]]) -> str:
        demo_blocks = "\n\n".join([f"Diff:\n{d}\nMessage:\n{m}" for d, m in demos])
        parts = []
        if demo_blocks:
            parts.append(f"Here are {len(demos)} examples of (Diff → Message):\n\n{demo_blocks}\n")
        parts.append(
            "Rules:\n"
            "- Write a single commit message (not a summary of the entire PR).\n"
            "- Use ONLY facts and identifiers that appear in the diff text.\n"
            "- Do not use external knowledge or cross-reference anything outside the diff.\n"
            "- Demos are examples only; do NOT reuse any facts or identifiers from them unless they appear in this diff.\n"
            "- If a rationale phrase appears in added comments/docs/tests, you may include one short clause using that exact wording.\n"
            "- You may infer brief intent only when it is directly implied by identifiers or added comments/docs/tests; do not introduce new facts.\n"
            "- Do NOT add extra details, causes, or implications.\n"
            "- Mention at most 1–2 key files/symbols; do not list multiple paths.\n"
            "- Prefer component-level wording over file lists.\n"
            "- Keep under ~30 words, imperative mood, no trailing period.\n\n"
            "Return JSON only with keys: message.\n"
            "- `message`: the commit message string.\n"
            f"Now write a concise, informative message for this diff:\n{diff_text}"
        )
        return "\n".join(parts)

    def _format_batch_prompt(self, items: List[Dict[str, Any]]) -> str:
        blocks = []
        for item in items:
            demos = item.get("demos") or []
            demo_blocks = "\n\n".join([f"Diff:\n{d}\nMessage:\n{m}" for d, m in demos])
            block = (
                "Commit:\n"
                f"SHA: {item['sha']}\n"
                f"Diff:\n{item['diff']}\n"
            )
            if demo_blocks:
                block += f"Demos:\n{demo_blocks}\n"
            blocks.append(block)
        payload = "\n\n---\n\n".join(blocks)
        return (
            "Rewrite each commit message using ONLY the provided diff.\n"
            "If the original message is already an accurate depiction of the diff, you may return it unchanged.\n"
            "Write a single commit message per item (not a PR summary).\n"
            "Use ONLY facts and identifiers that appear in the diff; do NOT add extra details.\n"
            "Do not use external knowledge or cross-reference anything outside the diff.\n"
            "Demos are examples only; do NOT reuse any facts or identifiers from them unless they appear in the current diff.\n"
            "If a rationale phrase appears in added comments/docs/tests, you may include one short clause using that exact wording.\n"
            "You may infer brief intent only when it is directly implied by identifiers or added comments/docs/tests; do not introduce new facts.\n"
            "Do NOT combine changes from other commits.\n"
            "Mention at most 1–2 key files/symbols; do not list multiple paths.\n"
            "Prefer component-level wording over file lists.\n"
            "Keep under ~30 words, imperative mood, no trailing period.\n"
            "Return JSON array with objects: {sha, message, changed}.\n\n"
            f"{payload}"
        )

    def _generate_with_icl(
        self,
        diff_text: str,
        repo_name: str,
        pr_number: int,
        files_touched: int | None,
    ) -> str:
        effective_k = self._effective_k()
        demos = self.demo_pool.top_k(
            diff_text,
            k=effective_k,
            scope=self.demo_scope,
            repo_name=repo_name,
            pr_number=pr_number,
            files_touched=files_touched,
        )
        if self.log_enabled:
            _log(
                f"[CMG][ICL] retriever={self.demo_pool.name} k={effective_k} "
                f"corpus_size={self.demo_pool.corpus_size} retrieved={len(demos)}"
        )
        prompt = self._format_prompt(diff_text, demos)
        raw = self.llm.chat(
            system_prompt=(
                "You are a senior software engineer. Write high-quality, factual commit messages "
                "from the provided diffs only. Do not invent facts or use external knowledge."
            ),
            user_prompt=prompt,
            log_type="cmg_icl",
            repo=repo_name,
            pr_number=pr_number,
        ).strip()
        parsed = self._parse_json_object(raw)
        if parsed and parsed.get("message"):
            return parsed
        return {"message": raw.strip()}

    def _parse_json_object(self, raw: str) -> Dict[str, Any] | None:
        raw = (raw or "").strip()
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return None

    def _generate_with_icl_batch(
        self,
        items: List[Dict[str, Any]],
        repo_name: str,
        pr_number: int,
    ) -> List[Dict[str, Any]]:
        prompt = self._format_batch_prompt(items)
        raw = self.llm.chat(
            system_prompt=(
                "You are a senior software engineer. Write high-quality, factual commit messages from diffs. "
                "Use ONLY the supplied diff text. Do not invent facts or use external knowledge."
            ),
            user_prompt=prompt,
            log_type="cmg_icl_batch",
            repo=repo_name,
            pr_number=pr_number,
        ).strip()
        parsed = self._parse_json_array(raw)
        if not parsed:
            return []
        return parsed

    def _parse_json_array(self, raw: str) -> List[Dict[str, Any]]:
        raw = (raw or "").strip()
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and isinstance(data.get("results"), list):
                return data["results"]
        except Exception:
            pass

        # Try to extract first [...] block
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start:end + 1]
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return []

    def _chunk_items(self, items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        chunks: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_tokens = 0
        max_commits = None
        if self.max_commits_per_chunk is not None:
            try:
                max_commits = int(self.max_commits_per_chunk)
            except (TypeError, ValueError):
                max_commits = None

        for item in items:
            item_tokens = self._estimate_tokens(item.get("prompt_text", ""))
            if current and (
                (current_tokens + item_tokens > self.max_chunk_tokens)
                or (max_commits is not None and len(current) >= max_commits)
            ):
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(item)
            current_tokens += item_tokens

        if current:
            chunks.append(current)
        return chunks

    def _batch_judge(self, items: List[Dict[str, Any]], repo_name: str, pr_number: int) -> Dict[str, bool]:
        if not self.qa.judge_enabled:
            return {item["sha"]: True for item in items}

        mode = "pairwise" if self.qa.pairwise else "binary"
        prompt_items = [
            {
                "sha": item["sha"],
                "diff": item["diff"],
                "original_message": item["original_message"],
                "candidate_message": item["candidate_message"],
            }
            for item in items
        ]
        user_prompt = (
            "You are a strict reviewer. Decide if the candidate commit message is accurate "
            "and clearly better for the provided diff.\n"
            "Return JSON array with objects:\n"
            "- binary: {sha, accept: true|false, reason}\n"
            "- pairwise: {sha, winner: \"original\"|\"candidate\", reason}\n\n"
            f"Mode: {mode}\n\n"
            f"Items:\n{json.dumps(prompt_items, indent=2)}"
        )
        raw = self.llm.chat(
            system_prompt=(
                "You are a senior software engineer and strict reviewer. "
                "Judge commit messages only using the provided diff and messages. "
                "Do not invent facts."
            ),
            user_prompt=user_prompt,
            log_type="cmg_judge_batch",
            repo=repo_name,
            pr_number=pr_number,
        ).strip()
        parsed = self._parse_json_array(raw)
        verdicts: Dict[str, bool] = {}
        for row in parsed or []:
            sha = row.get("sha")
            if not sha:
                continue
            if mode == "pairwise":
                verdicts[sha] = (row.get("winner") == "candidate")
            else:
                verdicts[sha] = bool(row.get("accept"))
        return verdicts

    def rewrite_if_needed(
        self,
        commits: List[Dict],
        repo_name: str,
        pr_number: int,
        selected_shas: set[str] | None = None,
    ) -> List[Dict]:
        resolved: Dict[str, str] = {}
        decisions: Dict[str, Dict[str, Any]] = {}
        candidates: List[Dict[str, Any]] = []

        for c in commits:
            msg = (c.get("message") or "").strip()
            sha = c.get("sha", "")
            patches = c.get("patches", [])
            diff = _build_diff(patches)
            has_patch = bool(patches)

            if selected_shas is not None and sha and sha not in selected_shas:
                if self.log_enabled:
                    _log(f"[CMG][KEEP]  {repo_name}#{pr_number} {sha} | reason=rank_skip")
                resolved[sha] = msg
                decisions[sha] = {
                    "sha": sha,
                    "original_message": msg,
                    "candidate_message": None,
                    "final_message": msg,
                    "status": "kept",
                    "reason": "rank_skip",
                    "patch_available": has_patch,
                }
                continue

            # Never rewrite merges/reverts
            if is_merge_or_revert(msg):
                if self.log_enabled:
                    _log(f"[CMG][KEEP]  {repo_name}#{pr_number} {sha} | reason=merge_or_revert")
                resolved[sha] = msg
                decisions[sha] = {
                    "sha": sha,
                    "original_message": msg,
                    "candidate_message": None,
                    "final_message": msg,
                    "status": "kept",
                    "reason": "merge_or_revert",
                    "patch_available": has_patch,
                }
                continue

            # If no diff or clearly fine, keep
            if not diff:
                reason = "patch_unavailable" if not has_patch else "quality_ok"
                if self.log_enabled:
                    _log(f"[CMG][KEEP]  {repo_name}#{pr_number} {sha} | reason={reason}")
                resolved[sha] = msg
                decisions[sha] = {
                    "sha": sha,
                    "original_message": msg,
                    "candidate_message": None,
                    "final_message": msg,
                    "status": "kept",
                    "reason": reason,
                    "patch_available": has_patch,
                }
                continue
            if not needs_cmg(msg):
                is_good, score, feats = self.qa.is_good_for_diff(msg, diff)
                if self.log_enabled:
                    _log(
                        f"[CMG][QA] original_sem={feats.get('sem', 0.0):.2f} "
                        f"original_score={score:.2f} "
                        f"good_threshold={self.qa.good_threshold:.2f}"
                    )
                if is_good:
                    if self.log_enabled:
                        _log(f"[CMG][KEEP]  {repo_name}#{pr_number} {sha} | reason=quality_ok")
                    resolved[sha] = msg
                    decisions[sha] = {
                        "sha": sha,
                        "original_message": msg,
                        "candidate_message": None,
                        "final_message": msg,
                        "status": "kept",
                        "reason": "quality_ok",
                        "patch_available": has_patch,
                    }
                    continue

            candidates.append(
                {
                    "sha": sha,
                    "original_message": msg,
                    "diff": diff,
                    "files_touched": len(c.get("files_touched") or []),
                    "patch_available": has_patch,
                }
            )

        if not candidates:
            out: List[Dict[str, Any]] = []
            for c in commits:
                sha = c.get("sha", "")
                msg = resolved.get(sha, c.get("message") or "")
                decision = decisions.get(sha) or {
                    "sha": sha,
                    "original_message": c.get("message") or "",
                    "candidate_message": None,
                    "final_message": msg,
                    "status": "kept",
                    "reason": "quality_ok",
                    "patch_available": bool(c.get("patches")),
                }
                out.append({"sha": sha, "message": msg, **decision})
            return out

        # Batch rewrite path (default)
        if self.batch_enabled:
            demo_k = self._effective_k()
            if self.batch_demo_k is not None:
                demo_k = min(demo_k, int(self.batch_demo_k))
            items: List[Dict[str, Any]] = []
            for cand in candidates:
                demos = self.demo_pool.top_k(
                    cand["diff"],
                    k=demo_k,
                    scope=self.demo_scope,
                    repo_name=repo_name,
                    pr_number=pr_number,
                    files_touched=cand.get("files_touched"),
                )
                item = {
                    **cand,
                    "demos": demos,
                }
                item["prompt_text"] = self._format_batch_prompt([item])
                items.append(item)

            chunks = self._chunk_items(items)
            if self.log_enabled:
                _log(f"[CMG][BATCH] chunks={len(chunks)} max_tokens={self.max_chunk_tokens}")

            rewritten_by_sha: Dict[str, str] = {}
            for chunk in chunks:
                batch_payload = [
                    {
                        "sha": i["sha"],
                        "original_message": i["original_message"],
                        "diff": i["diff"],
                        "demos": i.get("demos") or [],
                    }
                    for i in chunk
                ]
                results = self._generate_with_icl_batch(batch_payload, repo_name=repo_name, pr_number=pr_number)
                for row in results or []:
                    sha = row.get("sha")
                    if not sha:
                        continue
                    message = _postprocess_line(row.get("message") or "")
                    if not message:
                        continue
                    rewritten_by_sha[sha] = message

            to_judge: List[Dict[str, Any]] = []
            for cand in candidates:
                sha = cand["sha"]
                msg = cand["original_message"]
                diff = cand["diff"]
                hyp = rewritten_by_sha.get(sha, "")

                if not hyp:
                    if self.log_enabled:
                        _log(f"[CMG][KEEP]  {repo_name}#{pr_number} {sha} | reason=empty_hypothesis")
                    resolved[sha] = msg
                    decisions[sha] = {
                        "sha": sha,
                        "original_message": msg,
                        "candidate_message": None,
                        "final_message": msg,
                        "status": "kept",
                        "reason": "empty_hypothesis",
                        "patch_available": cand.get("patch_available", False),
                    }
                    continue

                if self.log_enabled:
                    _log(f"[CMG][CAND] {repo_name}#{pr_number} {sha}\n"
                          f"  ORIGINAL: {msg!r}\n"
                          f"  CANDIDATE: {hyp!r}")

                if not self._grounded_in_diff(hyp, diff):
                    if self.log_enabled:
                        _log(f"[CMG][KEEP]  {repo_name}#{pr_number} {sha} | reason=not_grounded")
                    resolved[sha] = msg
                    decisions[sha] = {
                        "sha": sha,
                        "original_message": msg,
                        "candidate_message": hyp,
                        "final_message": msg,
                        "status": "kept",
                        "reason": "not_grounded",
                        "patch_available": cand.get("patch_available", False),
                    }
                    continue
                if not self._strict_grounding_ok(hyp, diff):
                    if self.log_enabled:
                        _log(f"[CMG][KEEP]  {repo_name}#{pr_number} {sha} | reason=strict_grounding_fail")
                    resolved[sha] = msg
                    decisions[sha] = {
                        "sha": sha,
                        "original_message": msg,
                        "candidate_message": hyp,
                        "final_message": msg,
                        "status": "kept",
                        "reason": "strict_grounding_fail",
                        "patch_available": cand.get("patch_available", False),
                    }
                    continue

                accept_heur, dbg = self.qa.accept(msg, hyp, diff, allow_judge=False)
                if not accept_heur:
                    if self.log_enabled:
                        reason = dbg.get("rule", "no_benefit")
                        _log(f"[CMG][KEEP]  {repo_name}#{pr_number} {sha} | reason={reason}")
                    resolved[sha] = msg
                    decisions[sha] = {
                        "sha": sha,
                        "original_message": msg,
                        "candidate_message": hyp,
                        "final_message": msg,
                        "status": "kept",
                        "reason": dbg.get("rule", "reject"),
                        "patch_available": cand.get("patch_available", False),
                    }
                    continue

                to_judge.append(
                    {
                        "sha": sha,
                        "diff": diff,
                        "original_message": msg,
                        "candidate_message": hyp,
                    }
                )

            verdicts = self._batch_judge(to_judge, repo_name=repo_name, pr_number=pr_number)
            verdicts = verdicts or {}
            for item in to_judge:
                sha = item["sha"]
                msg = item["original_message"]
                hyp = item["candidate_message"]
                accept = verdicts.get(sha, False)
                if accept and hyp.lower() != msg.lower():
                    if self.log_enabled:
                        _log(f"[CMG][REWRITE] {repo_name}#{pr_number} {sha}\n"
                              f"  BEFORE: {msg!r}\n"
                              f"  AFTER : {hyp!r}")
                    resolved[sha] = hyp
                    decisions[sha] = {
                        "sha": sha,
                        "original_message": msg,
                        "candidate_message": hyp,
                        "final_message": hyp,
                        "status": "rewritten",
                        "reason": "heuristic_improve",
                        "patch_available": True,
                    }
                else:
                    if self.log_enabled:
                        _log(f"[CMG][KEEP]  {repo_name}#{pr_number} {sha} | reason=judge_reject")
                    resolved[sha] = msg
                    decisions[sha] = {
                        "sha": sha,
                        "original_message": msg,
                        "candidate_message": hyp,
                        "final_message": msg,
                        "status": "kept",
                        "reason": "judge_reject",
                        "patch_available": True,
                    }

        # Legacy per-commit path
        else:
            for c in candidates:
                sha = c["sha"]
                msg = c["original_message"]
                diff = c["diff"]
                files_touched = c["files_touched"]
                hyp_data = self._generate_with_icl(
                    diff,
                    repo_name=repo_name,
                    pr_number=pr_number,
                    files_touched=files_touched,
                )
                hyp = _postprocess_line(hyp_data.get("message") or "")

                if not hyp:
                    if self.log_enabled:
                        _log(f"[CMG][KEEP]  {repo_name}#{pr_number} {sha} | reason=empty_hypothesis")
                    resolved[sha] = msg
                    decisions[sha] = {
                        "sha": sha,
                        "original_message": msg,
                        "candidate_message": None,
                        "final_message": msg,
                        "status": "kept",
                        "reason": "empty_hypothesis",
                        "patch_available": c.get("patch_available", False),
                    }
                    continue

                if self.log_enabled:
                    _log(f"[CMG][CAND] {repo_name}#{pr_number} {sha}\n"
                          f"  ORIGINAL: {msg!r}\n"
                          f"  CANDIDATE: {hyp!r}")

                if not self._grounded_in_diff(hyp, diff):
                    if self.log_enabled:
                        _log(f"[CMG][KEEP]  {repo_name}#{pr_number} {sha} | reason=not_grounded")
                    resolved[sha] = msg
                    decisions[sha] = {
                        "sha": sha,
                        "original_message": msg,
                        "candidate_message": hyp,
                        "final_message": msg,
                        "status": "kept",
                        "reason": "not_grounded",
                        "patch_available": c.get("patch_available", False),
                    }
                    continue
                if not self._strict_grounding_ok(hyp, diff):
                    if self.log_enabled:
                        _log(f"[CMG][KEEP]  {repo_name}#{pr_number} {sha} | reason=strict_grounding_fail")
                    resolved[sha] = msg
                    decisions[sha] = {
                        "sha": sha,
                        "original_message": msg,
                        "candidate_message": hyp,
                        "final_message": msg,
                        "status": "kept",
                        "reason": "strict_grounding_fail",
                        "patch_available": c.get("patch_available", False),
                    }
                    continue

                accept, dbg = self.qa.accept(msg, hyp, diff, allow_judge=True)
                if accept and hyp.lower() != msg.lower():
                    if self.log_enabled:
                        _log(f"[CMG][REWRITE] {repo_name}#{pr_number} {sha}\n"
                              f"  BEFORE: {msg!r}\n"
                              f"  AFTER : {hyp!r}")
                    resolved[sha] = hyp
                    decisions[sha] = {
                        "sha": sha,
                        "original_message": msg,
                        "candidate_message": hyp,
                        "final_message": hyp,
                        "status": "rewritten",
                        "reason": dbg.get("rule", "heuristic_improve"),
                        "patch_available": c.get("patch_available", False),
                    }
                else:
                    if self.log_enabled:
                        reason = dbg.get("rule", "no_benefit")
                        _log(f"[CMG][KEEP]  {repo_name}#{pr_number} {sha} | reason={reason}")
                    resolved[sha] = msg
                    decisions[sha] = {
                        "sha": sha,
                        "original_message": msg,
                        "candidate_message": hyp,
                        "final_message": msg,
                        "status": "kept",
                        "reason": dbg.get("rule", "reject"),
                        "patch_available": c.get("patch_available", False),
                    }

        out: List[Dict[str, Any]] = []
        for c in commits:
            sha = c.get("sha", "")
            msg = resolved.get(sha, c.get("message") or "")
            decision = decisions.get(sha) or {
                "sha": sha,
                "original_message": c.get("message") or "",
                "candidate_message": None,
                "final_message": msg,
                "status": "kept",
                "reason": "quality_ok",
                "patch_available": bool(c.get("patches")),
            }
            out.append({"sha": sha, "message": msg, **decision})
        if self.log_enabled:
            original_by_sha = {c.get("sha", ""): (c.get("message") or "") for c in commits}
            rewrote = sum(1 for c in out if c["message"] != (original_by_sha.get(c["sha"]) or ""))
            _log(f"[CMG][SUMMARY] {repo_name}#{pr_number} rewrote {rewrote}/{len(commits)} commits")
        return out
