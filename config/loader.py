"""
Utilities for loading pipeline configuration (LLM, CMG, generation modes, etc.)
from config/pipeline.yaml with sane defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import copy
import yaml


DEFAULT_GENERATION_MODES: List[Dict[str, Any]] = [
    {
        "name": "raw",
        "use_cmg": False,
        "include_file_summaries": False,
        "description": "Original commit messages only; no CMG or file summaries.",
    },
    {
        "name": "cmg_only",
        "use_cmg": True,
        "include_file_summaries": False,
        "description": "CMG rewritten commits without file-level summaries.",
    },
    {
        "name": "file_summaries_only",
        "use_cmg": False,
        "include_file_summaries": True,
        "description": "Original commit messages plus file-level summaries.",
    },
    {
        "name": "full",
        "use_cmg": True,
        "include_file_summaries": True,
        "description": "CMG rewritten commits and file-level summaries.",
    },
]

DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": {
        "csv_path": "data/parsed.csv",
        "name": "parsed",
    },
    "llm": {
        "provider": "openai",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": None,
        "temperature": 0.2,
        "log_prompts": True,
    },
    "cmg": {
        "enabled": False,
        "graph_path": "results/knowledge_graph/graph.json",
        "demo_scope": "global",
        "k": 16,
        "batch_enabled": True,
        "max_chunk_tokens": 16000,
        "batch_demo_k": 2,
        "max_commits_per_chunk": None,
        "sem_model": "sentence-transformers/all-MiniLM-L6-v2",
        "log": False,
        "debug_demos": 3,
        "qa": {
            "use_sem": True,
            "score_threshold": 0.55,
            "good_threshold": 0.55,
            "min_improve": 0,
            "llm_judge": False,
            "pairwise": False,
        },
    },
    "generation_modes": DEFAULT_GENERATION_MODES,
    "active_generation_modes": [],
    "judge": {
        "default_provider": "openai",
    },
    "file_summaries": {
        "batch_size_small": 2,
        "batch_size_large": 3,
    },
    "ranking": {
        "file": {
            "include_all_if_file_count_leq": 15,
            "top_k_large": 25,
            "weights": {
                "path_keyword": 2.0,
                "api_impact": 2.0,
                "test_path": 1.0,
                "ci_path": 1.0,
                "size_log": 0.4,
                "status_added": 0.3,
                "status_deleted": 0.3,
                "noise_penalty": 1.5,
            },
        },
        "commit": {
            "include_all_if_commit_count_leq": 8,
            "top_k_large": 6,
            "weights": {
                "impact": 0.65,
                "intent": 0.35,
                "issue_link": 1.5,
                "starts_with_verb": 1.0,
                "cmg_quality": 1.0,
                "cmg_identifier_overlap": 0.8,
                "short_penalty": 0.7,
            },
        },
    },
}


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_modes(modes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for mode in modes:
        name = (mode.get("name") or "").strip()
        if not name:
            continue
        slug = mode.get("slug") or _slugify(name)
        if slug in seen:
            continue
        seen.add(slug)
        cleaned.append(
            {
                "name": name,
                "slug": slug,
                "use_cmg": bool(mode.get("use_cmg", True)),
                "include_file_summaries": bool(mode.get("include_file_summaries", True)),
                "description": mode.get("description") or "",
            }
        )
    if not cleaned:
        cleaned = [
            {**mode, "slug": _slugify(mode["name"])} for mode in DEFAULT_GENERATION_MODES
        ]
    return cleaned


def _slugify(value: str) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "mode"


def load_pipeline_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML, merging with defaults.
    """
    root_dir = Path(__file__).resolve().parent.parent
    if path is None:
        cfg_path = root_dir / "config" / "pipeline.yaml"
    else:
        cfg_path = Path(path)
        if not cfg_path.is_absolute():
            cfg_path = root_dir / cfg_path
    user_config: Dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as handle:
            user_config = yaml.safe_load(handle) or {}
    merged = _deep_merge(DEFAULT_CONFIG, user_config)
    dataset = merged.get("dataset") or {}
    if isinstance(dataset, dict):
        name = (dataset.get("name") or "").strip()
        csv_path = (dataset.get("csv_path") or "").strip()
        if not name and csv_path:
            name = Path(csv_path).stem
        if name and not csv_path:
            csv_path = str(Path("data") / f"{name}.csv")
        if name:
            dataset["name"] = name
        if csv_path:
            dataset["csv_path"] = csv_path
        merged["dataset"] = dataset
    dataset_name = (merged.get("dataset") or {}).get("name") or ""
    cmg_cfg = merged.get("cmg") or {}
    if isinstance(cmg_cfg, dict) and dataset_name:
        graph_path = (cmg_cfg.get("graph_path") or "").strip()
        if not graph_path or graph_path == "results/knowledge_graph/graph.json":
            cmg_cfg["graph_path"] = f"results/knowledge_graph/graph-{dataset_name}.json"
        merged["cmg"] = cmg_cfg
    merged["generation_modes"] = _normalize_modes(merged.get("generation_modes", []))
    active = merged.get("active_generation_modes") or []
    if isinstance(active, list) and active:
        active_set = {str(a).strip().lower() for a in active if str(a).strip()}
        if active_set:
            merged["generation_modes"] = [
                mode for mode in merged["generation_modes"]
                if mode["name"].lower() in active_set or mode["slug"].lower() in active_set
            ] or merged["generation_modes"]
    return merged
