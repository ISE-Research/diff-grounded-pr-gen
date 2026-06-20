"""
Shared ranking helpers for files and commits.
Language-agnostic, deterministic scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import math
import re


PATH_KEYWORDS = [
    "api",
    "auth",
    "security",
    "config",
    "schema",
    "migration",
    "migrations",
    "permission",
    "permissions",
    "policy",
    "acl",
    "access",
    "role",
    "roles",
    "oauth",
    "jwt",
    "token",
    "session",
    "workflow",
    "ci",
    "pipeline",
    "build",
    "docker",
    "k8s",
    "kubernetes",
    "deploy",
    "release",
]

NOISE_PATH_PATTERNS = [
    r"/node_modules/",
    r"/.venv/",
    r"/.eggs/",
]

NOISE_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pipfile.lock",
    "gemfile.lock",
    "go.sum",
    "cargo.lock",
    "composer.lock",
}
NOISE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".svg",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".7z",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
    ".bin",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
}
DATA_DUMP_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
}
DATA_DUMP_PATH_PATTERNS = [
    r"/data/",
    r"/datasets/",
    r"/fixtures/",
]
API_PATTERN = re.compile(r"\b(public|export|class|interface|enum|record|def|function|type|schema)\b", re.IGNORECASE)


def _normalized_path(path: str) -> str:
    return (path or "").lower().replace("\\", "/")


def path_has_keyword(path: str) -> bool:
    lowered = _normalized_path(path)
    for keyword in PATH_KEYWORDS:
        pattern = rf"(?:^|/|_|-){re.escape(keyword)}(?:/|_|-|\\.|$)"
        if re.search(pattern, lowered):
            return True
    return False


def is_test_path(path: str) -> bool:
    lowered = _normalized_path(path)
    return (
        "/test/" in lowered
        or "/tests/" in lowered
        or "/spec/" in lowered
        or "/specs/" in lowered
        or "/__tests__/" in lowered
        or lowered.endswith("_test.py")
        or lowered.endswith("_spec.rb")
        or lowered.endswith(".test.js")
        or lowered.endswith(".spec.js")
        or lowered.endswith(".test.ts")
        or lowered.endswith(".spec.ts")
    )


def is_ci_path(path: str) -> bool:
    lowered = _normalized_path(path)
    return (
        "/.github/workflows/" in lowered
        or lowered.endswith("jenkinsfile")
        or lowered.endswith(".gitlab-ci.yml")
        or "/.circleci/" in lowered
        or lowered.endswith("dockerfile")
        or lowered.endswith("docker-compose.yml")
    )


def is_noise_path(path: str) -> bool:
    lowered = _normalized_path(path)
    filename = Path(lowered).name
    if filename in NOISE_FILENAMES:
        return True
    if ".min." in filename:
        return True
    if Path(lowered).suffix in NOISE_EXTENSIONS:
        return True
    if Path(lowered).suffix in DATA_DUMP_EXTENSIONS:
        for pattern in DATA_DUMP_PATH_PATTERNS:
            if re.search(pattern, lowered):
                return True
    for pattern in NOISE_PATH_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return False


def api_impact_from_patch(patch: str) -> bool:
    if not patch:
        return False
    for line in patch.splitlines():
        if line.startswith(("diff --git", "@@", "+++", "---")):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            if API_PATTERN.search(line[1:]):
                return True
        if line.startswith("-") and not line.startswith("---"):
            if API_PATTERN.search(line[1:]):
                return True
    return False


def compute_file_scores(
    files: List[Dict[str, Any]],
    weights: Dict[str, float],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for f in files or []:
        filename = f.get("filename") or ""
        if not filename:
            continue
        patch = (f.get("patch") or "").strip()
        api_impact = api_impact_from_patch(patch)
        features = {
            "changes": f.get("changes"),
            "status": f.get("status"),
            "is_api_impact": api_impact,
            "path_keyword": path_has_keyword(filename),
            "is_test": is_test_path(filename),
            "is_ci": is_ci_path(filename),
            "is_noise": is_noise_path(filename),
        }
        scores[filename] = file_score(features, weights)
    return scores


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def file_score(features: Dict[str, Any], weights: Dict[str, float]) -> float:
    """
    Compute a file impact score using language-agnostic signals.
    features:
      - changes, status, is_api_impact, path_keyword, is_test, is_ci, is_noise
    """
    score = 0.0
    changes = _safe_float(features.get("changes"), 0.0)
    score += weights.get("size_log", 0.0) * math.log1p(max(0.0, changes))

    if features.get("path_keyword"):
        score += weights.get("path_keyword", 0.0)
    if features.get("is_api_impact"):
        score += weights.get("api_impact", 0.0)
    if features.get("is_test"):
        score += weights.get("test_path", 0.0)
    if features.get("is_ci"):
        score += weights.get("ci_path", 0.0)

    status = (features.get("status") or "").lower()
    if status in {"added", "new"}:
        score += weights.get("status_added", 0.0)
    if status in {"removed", "deleted"}:
        score += weights.get("status_deleted", 0.0)

    if features.get("is_noise"):
        score -= weights.get("noise_penalty", 0.0)
    return score


def select_top_files(
    scored: List[Dict[str, Any]],
    max_files: int,
    always_include: Iterable[str],
) -> List[Dict[str, Any]]:
    """
    scored: list of {filename, score, ...}
    always_include: filenames to force-include.
    """
    include_set = {f for f in always_include if f}
    forced = [f for f in scored if f.get("filename") in include_set]
    remainder = [f for f in scored if f.get("filename") not in include_set]
    remainder.sort(key=lambda f: f.get("score", 0.0), reverse=True)

    picked = forced
    if len(picked) < max_files:
        picked.extend(remainder[: max(0, max_files - len(picked))])
    else:
        picked = picked[:max_files]
    return picked


def commit_score(
    commit: Dict[str, Any],
    file_scores: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """
    Compute commit score from file impact + intent signals.
    """
    files = commit.get("files_touched") or []
    if not files and commit.get("patches"):
        files = [p.get("filename") for p in commit.get("patches") or [] if p.get("filename")]
    impact = sum(file_scores.get(f, 0.0) for f in files or [])

    has_issue = bool(commit.get("cmg_issue_refs"))
    starts_with_verb = bool(commit.get("starts_with_verb"))
    is_short = bool(commit.get("is_short"))
    cmg_quality = _safe_float(commit.get("cmg_quality_score"), 0.0)
    cmg_identifier_overlap = 1.0 if commit.get("cmg_identifier_overlap") else 0.0

    intent = 0.0
    if has_issue:
        intent += weights.get("issue_link", 0.0)
    if starts_with_verb:
        intent += weights.get("starts_with_verb", 0.0)
    intent += weights.get("cmg_quality", 0.0) * cmg_quality
    intent += weights.get("cmg_identifier_overlap", 0.0) * cmg_identifier_overlap
    if is_short:
        intent -= weights.get("short_penalty", 0.0)

    return (weights.get("impact", 0.65) * impact) + (weights.get("intent", 0.35) * intent)


def rank_commits(
    commits: List[Dict[str, Any]],
    file_scores: Dict[str, float],
    weights: Dict[str, float],
) -> List[Tuple[str, float]]:
    scored: List[Tuple[str, float]] = []
    for commit in commits or []:
        sha = commit.get("sha")
        if not sha:
            continue
        score = commit_score(commit, file_scores, weights)
        scored.append((sha, score))
    scored.sort(key=lambda row: row[1], reverse=True)
    return scored
