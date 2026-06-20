from __future__ import annotations

import re
import math
from typing import Dict, Tuple, Optional

def _log(msg: str) -> None:
    print(msg)
    print()

# --------------------------------------------
# Basic heuristics used standalone as a fast gate
# --------------------------------------------

_BAD_WEAK = {"wip", "tmp", "fix", "update", "misc", "minor", "changes", "stuff"}
_MERGE_RE = re.compile(r"^\s*(merge|revert)\b", re.I)
_WS_SPLIT = re.compile(r"[^\w./#:+-]+")
_ISSUE_REF_RE = re.compile(r"(?:\b[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#\d+\b|#\d+)")


def _extract_issue_refs(msg: str) -> list[str]:
    if not msg:
        return []
    refs = _ISSUE_REF_RE.findall(msg)
    seen = set()
    ordered: list[str] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            ordered.append(ref)
    return ordered

def is_merge_or_revert(msg: Optional[str]) -> bool:
    if not msg:
        return False
    return bool(_MERGE_RE.match(msg.strip()))

def is_good_commit_message(msg: Optional[str]) -> bool:
    """
    Define a "good" commit message:
    - Not a merge/revert
    - Imperative verb start
    - 3-14 tokens, 8-100 chars
    - Avoids generic/vague words
    """
    if not msg:
        return False
    msg = msg.strip()
    if not msg:
        return False
    if is_merge_or_revert(msg):
        return False
    if any(tok in msg.lower() for tok in _BAD_WEAK):
        return False
    if not (8 <= len(msg) <= 100):
        return False
    tokens = _tokens(msg)
    if not (3 <= len(tokens) <= 14):
        return False
    if _imperative_score(msg) <= 0.0:
        return False
    return True

def needs_cmg(msg: Optional[str]) -> bool:
    """
    True -> consider generating a better message.
    We never rewrite merges/reverts; very short messages are candidates.
    """
    if is_merge_or_revert(msg):
        return False  # preserve merges/reverts for traceability
    return not is_good_commit_message(msg)


def compute_commit_quality_annotations(
    msg: Optional[str],
    patches: Optional[list[dict]] = None,
    files_touched: Optional[list[str]] = None,
) -> Dict[str, object]:
    """
    Compute lightweight, deterministic quality signals for a commit message.
    Intended for graph annotations (no LLM, no embeddings).
    """
    message = (msg or "").strip()
    is_merge = is_merge_or_revert(message)
    is_good = is_good_commit_message(message)

    msg_len = len(message)
    tokens = _tokens(message)
    token_count = len(tokens)
    starts_with_verb = _imperative_score(message) > 0.0

    has_identifier = False
    has_path_or_ext = False
    for t in tokens:
        if (
            "/" in t or "\\" in t or "." in t
            or "_" in t or re.search(r"[A-Z][a-z]+[A-Z]", t)
            or re.search(r"\d", t)
        ):
            has_identifier = True
        if "/" in t or "." in t:
            has_path_or_ext = True
    has_issue_ref = bool(_ISSUE_REF_RE.search(message))
    issue_refs = _extract_issue_refs(message)

    add_lines = 0
    del_lines = 0
    diff_lines: list[str] = []
    for p in patches or []:
        patch = p.get("patch") or ""
        for line in patch.splitlines():
            if line.startswith(("diff --git", "@@", "+++", "---")):
                continue
            if line.startswith("+"):
                add_lines += 1
                diff_lines.append(line[1:])
            elif line.startswith("-"):
                del_lines += 1
                diff_lines.append(line[1:])
    diff_tokens = _tokens(" ".join(diff_lines))
    msg_set = set(tokens)
    diff_set = set(diff_tokens)
    jac = _jaccard(msg_set, diff_set)
    tfidf = _tfidf_cosine(tokens, diff_tokens)
    sem = _semantic_sim(message, " ".join(diff_lines), "sentence-transformers/all-MiniLM-L6-v2")
    overlap_tokens = [t for t in tokens if t in diff_tokens]
    overlap_bonus = min(len(overlap_tokens), 20) / 20.0
    quality_score = (
        (0.20 * jac)
        + (0.15 * tfidf)
        + (0.55 * sem)
        + (0.10 * overlap_bonus)
    )
    quality_score = max(0.0, min(1.0, quality_score))
    diff_identifiers = {
        t for t in diff_tokens
        if (
            "/" in t or "\\" in t or "." in t
            or "_" in t or re.search(r"[A-Z][a-z]+[A-Z]", t)
            or re.search(r"\d", t)
        )
    }
    msg_identifiers = {
        t for t in tokens
        if (
            "/" in t or "\\" in t or "." in t
            or "_" in t or re.search(r"[A-Z][a-z]+[A-Z]", t)
            or re.search(r"\d", t)
        )
    }
    identifier_overlap = bool(diff_identifiers & msg_identifiers)

    if diff_identifiers and not identifier_overlap:
        is_good = False
        rule = "not_grounded"
    elif is_good:
        rule = "good"
    elif is_merge:
        rule = "merge_or_revert"
    elif msg_len < 8 or token_count < 3:
        rule = "too_short"
    elif msg_len > 100 or token_count > 14:
        rule = "too_long"
    elif not starts_with_verb:
        rule = "not_imperative"
    elif any(tok in message.lower() for tok in _BAD_WEAK):
        rule = "vague_word"
    else:
        rule = "needs_rewrite"

    files_count = len(files_touched or [])
    has_tests = any(
        "/test" in f.lower() or "/tests" in f.lower() or f.lower().startswith("test_")
        for f in (files_touched or [])
    )

    return {
        "cmg_is_good": is_good,
        "cmg_needs_rewrite": (not is_merge) and (not is_good),
        "cmg_rule": rule,
        "cmg_msg_len": msg_len,
        "cmg_token_count": token_count,
        "cmg_starts_with_verb": starts_with_verb,
        "cmg_is_merge_or_revert": is_merge,
        "cmg_has_identifier": has_identifier,
        "cmg_has_path_or_ext": has_path_or_ext,
        "cmg_has_issue_ref": has_issue_ref,
        "cmg_identifier_overlap": identifier_overlap,
        "cmg_issue_refs": issue_refs,
        "cmg_has_tests": has_tests,
        "cmg_files_touched": files_count,
        "cmg_add_lines": add_lines,
        "cmg_del_lines": del_lines,
        "cmg_diff_token_count": len(diff_tokens),
        "cmg_sem_score": round(sem, 4),
        "cmg_overlap_bonus": round(overlap_bonus, 4),
        "cmg_quality_score": round(quality_score, 4),
    }

def _tokens(txt: str) -> list[str]:
    if not txt:
        return []
    return [t.lower() for t in _WS_SPLIT.split(txt) if t]

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union

def _tfidf_cosine(a_tokens: list[str], b_tokens: list[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    vocab: dict[str, int] = {}
    for t in a_tokens + b_tokens:
        if t not in vocab:
            vocab[t] = len(vocab)

    def _tf(tokens: list[str]) -> list[float]:
        vec = [0.0] * len(vocab)
        for t in tokens:
            vec[vocab[t]] += 1.0
        if not tokens:
            return vec
        return [v / len(tokens) for v in vec]

    def _df(tokens: list[str]) -> set[str]:
        return set(tokens)

    df = [0] * len(vocab)
    for t in _df(a_tokens):
        df[vocab[t]] += 1
    for t in _df(b_tokens):
        df[vocab[t]] += 1

    idf = []
    for count in df:
        idf.append(math.log((2 + 1) / (count + 1)) + 1.0)

    a_tf = _tf(a_tokens)
    b_tf = _tf(b_tokens)
    a_vec = [a_tf[i] * idf[i] for i in range(len(vocab))]
    b_vec = [b_tf[i] * idf[i] for i in range(len(vocab))]

    num = sum(a_vec[i] * b_vec[i] for i in range(len(vocab)))
    a_den = sum(v * v for v in a_vec) ** 0.5
    b_den = sum(v * v for v in b_vec) ** 0.5
    if a_den == 0 or b_den == 0:
        return 0.0
    return num / (a_den * b_den)

def _extract_anchors(diff: str, max_anchors: int = 5) -> list[str]:
    anchors: list[str] = []
    seen = set()
    stop = {
        "add", "adds", "added", "fix", "fixes", "fixed", "update", "updates", "updated",
        "refactor", "remove", "removed", "rename", "renamed", "revert", "improve",
        "optimize", "document", "bump", "implement", "enable", "disable", "handle",
        "support", "streamline", "migrate", "correct", "clean", "extract",
        "change", "changes", "adjust", "tweak", "misc",
    }
    for line in (diff or "").splitlines():
        if not line.startswith("+"):
            continue
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", line):
            lower = token.lower()
            if lower in stop:
                continue
            if token in seen:
                continue
            seen.add(token)
            anchors.append(lower)
            if len(anchors) >= max_anchors:
                return anchors
    return anchors

def _length_score(n: int) -> float:
    """
    Soft-spot around ~50 chars. Score in [0,1].
    """
    # triangular-ish preference: 0 at 0/100+, 1 near 50
    return max(0.0, 1.0 - abs(n - 50) / 50.0)

def _specificity_score(msg: str) -> float:
    """
    Crude proxy for 'specific': presence of identifiers, paths, code-ish tokens.
    """
    toks = _tokens(msg)
    if not toks:
        return 0.0
    code_like = 0
    for t in toks:
        if (
            "/" in t or "\\" in t or "." in t  # paths or file.ext
            or "_" in t or re.search(r"[A-Z][a-z]+[A-Z]", t)  # CamelCase-ish
            or re.search(r"\d", t)  # numbers in ids
        ):
            code_like += 1
    return min(1.0, code_like / max(3, len(toks)))

def _imperative_score(msg: str) -> float:
    """
    Very light heuristic: common commit-verb starts; penalize past-participles.
    """
    if not msg:
        return 0.0
    first = _tokens(msg[:40])
    if not first:
        return 0.0
    w = first[0]
    good_starts = {
        "add","fix","update","refactor","remove","rename","revert",
        "improve","optimize","document","bump","implement","enable","disable",
        "handle","support","streamline","migrate","correct","clean","extract"
    }
    bad_suffix = w.endswith("ed") or w.endswith("ing")
    if w in good_starts and not bad_suffix:
        return 1.0
    if not bad_suffix:
        return 0.5
    return 0.0

# --------------------------------------------
# Optional semantic similarity (Sentence-Transformers)
# --------------------------------------------
_ST = None
_np = None

def _maybe_load_st(model_name: str, log_enabled: bool = False):
    global _ST, _np
    if _ST is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        _ST = SentenceTransformer(model_name)
        _np = np
        if log_enabled:
            _log(f"[CMG][QA] loaded sentence-transformer model={model_name}")
    except Exception as e:
        if log_enabled:
            _log(f"[CMG][QA] WARN cannot load sentence-transformers: {e}")
        _ST = None
        _np = None

def _cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    denom = (float((a**2).sum()) ** 0.5) * (float((b**2).sum()) ** 0.5)
    if denom <= 0:
        return 0.0
    return float((a * b).sum()) / denom

def _semantic_sim(msg: str, diff: str, model_name: str, log_enabled: bool = False) -> float:
    if not msg or not diff:
        return 0.0
    _maybe_load_st(model_name, log_enabled=log_enabled)
    if _ST is None or _np is None:
        return 0.0
    # Truncate diff for speed; keep signal (added/deleted lines ideally)
    snippet = diff[:1200]
    emb = _ST.encode([msg, snippet], normalize_embeddings=False)
    return max(0.0, min(1.0, (1.0 + _cosine(_np.array(emb[0]), _np.array(emb[1]))) / 2.0))

# --------------------------------------------
# LLM-as-a-judge (optional)
# --------------------------------------------

import inspect

class _LLMHandle:
    """
    Adapts to either:
      - chat(system_prompt: str, user_prompt: str)  # your LLMClientWrapper style
      - chat(messages: list[{role, content}])        # OpenAI-style
    Returns the raw text string.
    """
    def __init__(self, llm_client, log_enabled: bool = False):
        self.llm = llm_client
        self.log_enabled = log_enabled

    def _chat(self, system: str, user: str) -> str:
        if self.llm is None or not hasattr(self.llm, "chat"):
            raise RuntimeError("LLM client has no .chat()")

        try:
            sig = inspect.signature(self.llm.chat)
        except (ValueError, TypeError):
            sig = None

        # Prefer named-parameter style if available
        if sig and ("system_prompt" in sig.parameters and "user_prompt" in sig.parameters):
            return self.llm.chat(system_prompt=system, user_prompt=user)

        # Else try positional (system, user)
        if sig and len(sig.parameters) >= 2:
            try:
                return self.llm.chat(system, user)
            except TypeError:
                pass

        # Else assume OpenAI-style messages=[...]
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": user}]
        return self.llm.chat(messages)

    def judge_pairwise(self, original: str, candidate: str, diff: str) -> Optional[bool]:
        if self.llm is None:
            return None
        system = (
            "You are a senior software reviewer. Compare two commit messages for the same code diff. "
            "Pick the BETTER one for traceability, specificity, and accuracy. Reply with one token: A or B."
        )
        user = (
            f"DIFF (truncated):\n{diff[:1500]}\n\n"
            f"A) {original}\n"
            f"B) {candidate}\n\n"
            "Which is better? Reply exactly with 'A' or 'B'."
        )
        try:
            txt = self._chat(system, user).strip().upper()
            if txt.startswith("A"):
                return False  # A (original) is better → reject candidate
            if txt.startswith("B"):
                return True   # B (candidate) is better → accept
        except Exception as e:
            if self.log_enabled:
                _log(f"[CMG][JUDGE] pairwise error: {e}")
        return None

    def judge_binary(self, original: str, candidate: str, diff: str) -> Optional[bool]:
        if self.llm is None:
            return None
        system = (
            "You are a senior software reviewer. Decide if the NEW commit message is CLEARLY better "
            "than the ORIGINAL for the provided code diff. Reply 'YES' if the new one is clearly better, "
            "otherwise reply 'NO'."
        )
        user = (
            f"DIFF (truncated):\n{diff[:1500]}\n\n"
            f"ORIGINAL: {original}\n"
            f"NEW: {candidate}\n\n"
            "Is NEW clearly better? Reply exactly 'YES' or 'NO'."
        )
        try:
            txt = self._chat(system, user).strip().upper()
            if txt.startswith("Y"):
                return True
            if txt.startswith("N"):
                return False
        except Exception as e:
            if self.log_enabled:
                _log(f"[CMG][JUDGE] binary error: {e}")
        return None

# --------------------------------------------
# Main quality gate class
# --------------------------------------------

class CmgQuality:
    """
    Hybrid quality scoring:
      score = w_len*len + w_spec*spec + w_jac*jac + w_imp*imp + w_sem*sem
    Gate:
      accept candidate iff  score(candidate) >= threshold  AND
                            score(candidate) - score(original) >= min_delta
    Optional LLM-as-judge can override when heuristic is inconclusive.
    """
    def __init__(
        self,
        llm_client=None,
        settings: Optional[Dict[str, Any]] = None,
        log_enabled: bool = False,
        sem_model_name: Optional[str] = None,
    ):
        settings = settings or {}
        self.log_enabled = log_enabled
        self.threshold = float(settings.get("score_threshold", 0.55))
        self.good_threshold = float(settings.get("good_threshold", self.threshold))
        self.min_delta = float(settings.get("min_improve", 0.05))
        self.min_delta_high_score = float(settings.get("min_improve_high_score", 0.35))
        self.min_delta_high = float(settings.get("min_improve_high", 0.02))
        self.use_sem = bool(settings.get("use_sem", True))

        self.judge_enabled = bool(settings.get("llm_judge", False))
        self.pairwise = bool(settings.get("pairwise", False))

        if sem_model_name:
            self.sem_model_name = sem_model_name
        else:
            self.sem_model_name = settings.get("sem_model") or "sentence-transformers/all-MiniLM-L6-v2"
        self._judge = _LLMHandle(llm_client, log_enabled=self.log_enabled) if self.judge_enabled else None

        # weights
        self.w_jac, self.w_tfidf = 0.20, 0.15
        self.w_sem = 0.55 if self.use_sem else 0.0
        self.w_overlap = 0.10
        self.w_anchor = 0.10

    def score(self, msg: str, diff: str) -> Tuple[float, Dict[str, float]]:
        msg = (msg or "").strip()
        diff = diff or ""

        msg_tokens = _tokens(msg)
        msg_set  = set(msg_tokens)
        # focus diff on code tokens from +/- lines if caller built that; otherwise whole diff text
        diff_tokens = _tokens(diff)
        diff_set = set(diff_tokens)
        jac   = _jaccard(msg_set, diff_set)
        tfidf = _tfidf_cosine(msg_tokens, diff_tokens)
        overlap_bonus = min(len([t for t in msg_tokens if t in diff_tokens]), 20) / 20.0
        anchors = _extract_anchors(diff)
        anchor_hits = min(len([t for t in msg_tokens if t in anchors]), 3)
        anchor_bonus = anchor_hits / 3.0 if anchors else 0.0

        sem   = _semantic_sim(msg, diff, self.sem_model_name, log_enabled=self.log_enabled) if self.use_sem else 0.0

        score = (
            self.w_jac * jac +
            self.w_tfidf * tfidf +
            self.w_sem * sem +
            self.w_overlap * overlap_bonus +
            self.w_anchor * anchor_bonus
        )
        score = max(0.0, min(1.0, score))
        feats = {
            "jac": round(jac, 2),
            "tfidf": round(tfidf, 2),
            "sem": round(sem, 2),
            "overlap": round(overlap_bonus, 2),
            "anchor": round(anchor_bonus, 2),
        }
        return score, feats

    def accept(self, orig_msg: str, cand_msg: str, diff: str, allow_judge: bool = True) -> Tuple[bool, Dict]:
        """
        Returns (accepted?, debug_info)
        """
        o_score, o_feats = self.score(orig_msg, diff)
        c_score, c_feats = self.score(cand_msg, diff)

        if self.log_enabled:
            _log(
                f"[CMG][QA] score={c_score:.2f} "
                f"jac={c_feats.get('jac', 0.0):.2f} "
                f"tfidf={c_feats.get('tfidf', 0.0):.2f} "
                f"sem={c_feats.get('sem', 0.0):.2f}"
            )

        effective_min_delta = self.min_delta_high if o_score >= self.min_delta_high_score else self.min_delta
        if c_score > o_score:
            return True, {"orig": o_score, "cand": c_score, "rule": "heuristic_improve"}

        # If heuristics say no, optionally consult LLM-as-judge
        if allow_judge and self.judge_enabled and self._judge is not None:
            mode = "pairwise" if self.pairwise else "binary"
            if self.log_enabled:
                _log(f"[CMG][JUDGE] mode={mode}")

            verdict = (self._judge.judge_pairwise(orig_msg, cand_msg, diff)
                       if self.pairwise else
                       self._judge.judge_binary(orig_msg, cand_msg, diff))

            if verdict is True:
                if self.log_enabled:
                    _log(f"[CMG][JUDGE] accepted by LLM ({mode})")
                return True, {"orig": o_score, "cand": c_score, "rule": f"llm_judge_accept:{mode}"}
            if verdict is False:
                if self.log_enabled:
                    _log(f"[CMG][JUDGE] rejected by LLM ({mode})")
                return False, {"orig": o_score, "cand": c_score, "rule": f"llm_judge_reject:{mode}"}

            # fallthrough when judge couldn't decide or errored
            if self.log_enabled:
                _log(f"[CMG][JUDGE] no decision ({mode})")

        return False, {"orig": o_score, "cand": c_score, "rule": "reject"}

    def is_good_for_diff(self, msg: str, diff: str) -> Tuple[bool, float, Dict[str, float]]:
        """
        Return True if a message is semantically aligned with the diff at or above
        the configured goodness threshold.
        """
        score, feats = self.score(msg, diff)
        return score >= self.good_threshold, score, feats
