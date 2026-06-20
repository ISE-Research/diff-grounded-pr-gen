#!/usr/bin/env python3
"""Analyze human survey rankings using the blinded CSV and researcher key.

Inputs:
- Human Evaluation Survey- Pull Request Descriptions.csv
- survey_researcher_key.md

Outputs (prefix default: human_evaluation_summary):
- <prefix>.json
- <prefix>.csv
- <prefix>.md
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent
RESEARCH_PAPER_DIR = REPO_ROOT / "research-paper"
PAPER_FINAL_RESULTS_DIR = RESEARCH_PAPER_DIR / "final-results"
HUMAN_DATA_DIR = ROOT / "data" / "human-data"
TABLES_DIR = ROOT / "eval" / "tables"
PLOTS_DIR = ROOT / "eval" / "plots"
DEFAULT_CSV = HUMAN_DATA_DIR / "Human Evaluation Survey- Pull Request Descriptions.csv"
DEFAULT_KEY = HUMAN_DATA_DIR / "survey_researcher_key.md"
DEFAULT_PREFIX = "human_evaluation_summary"
VERSION_LABELS = ("A", "B", "C", "D", "E")
MODE_ORDER = (
    "original",
    "generated_raw",
    "generated_cmg_only",
    "generated_file_summaries_only",
    "generated_full",
)
DATASET_ALIASES = {
    "parsed": "prsummarizer",
    "trudeau": "prsummarizer",
    "prsummarizer": "prsummarizer",
    "aidev": "aidev",
}
DATASET_LABELS = {
    "aidev": "AIDev",
    "prsummarizer": "PRSummarizer",
}
DATASET_ORDER = {"aidev": 0, "prsummarizer": 1}
GENERIC_REASON_TEXTS = {
    "no",
    "descriptive",
    "clear description",
    "understandable",
    "clean description",
    "clearly designed",
    "clearly described",
}
THEME_KEYWORDS = {
    "clarity_readability": [
        "clear",
        "clearly",
        "easy to read",
        "understand",
        "understandable",
        "readable",
        "comprehension",
    ],
    "conciseness": [
        "concise",
        "short",
        "to the point",
        "brief",
        "without unnecessary",
        "not wordy",
    ],
    "completeness_coverage": [
        "complete",
        "comprehensive",
        "covers",
        "everything",
        "full change",
        "scope",
        "impact",
    ],
    "structure_organization": [
        "structured",
        "well-structured",
        "organized",
        "format",
        "section",
    ],
    "specificity_concrete": [
        "specific",
        "exactly",
        "details",
        "concrete",
        "affected",
        "what changed",
        "why",
    ],
    "implementation_noise_negative": [
        "word vomit",
        "too much",
        "low-level",
        "implementation detail",
        "excessive detail",
    ],
    "commit_hash_negative": [
        "commit hash",
        "commit hashes",
        "sha",
        "metadata",
        "digging",
    ],
}


def parse_rank(text: str) -> int | None:
    if not text:
        return None
    m = re.match(r"\s*([1-5])", text)
    if not m:
        return None
    return int(m.group(1))


def normalize_dataset_name(dataset: str) -> str:
    return DATASET_ALIASES.get(dataset.lower(), dataset.lower())


def dataset_sort_key(dataset: str) -> Tuple[int, str]:
    return (DATASET_ORDER.get(dataset, 99), dataset)


def label_dataset(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def parse_researcher_key(path: Path) -> List[Dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    tasks: List[Dict[str, object]] = []
    current: Dict[str, object] | None = None

    for line in lines:
        header = re.match(r"##\s+([PA]\d+)\s+\((parsed|trudeau|aidev)\)", line.strip())
        if header:
            current = {
                "task_id": header.group(1),
                "dataset": normalize_dataset_name(header.group(2)),
                "versions": {},
            }
            tasks.append(current)
            continue

        version = re.match(r"- Version ([A-E]):\s+([a-z_]+)", line.strip())
        if version and current is not None:
            current["versions"][version.group(1)] = version.group(2)

    for task in tasks:
        versions = task["versions"]
        missing = [v for v in VERSION_LABELS if v not in versions]
        if missing:
            raise ValueError(f"Missing version mappings for {task['task_id']}: {missing}")

    return tasks


def mean(values: List[int]) -> float:
    return float(sum(values)) / float(len(values)) if values else 0.0


def safe_mode_sort_key(mode: str) -> Tuple[int, str]:
    if mode in MODE_ORDER:
        return (MODE_ORDER.index(mode), mode)
    return (len(MODE_ORDER), mode)


def classify_reason_themes(reason: str) -> List[str]:
    lowered = reason.lower()
    labels: List[str] = []
    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            labels.append(theme)
    return labels


def analyze(csv_path: Path, key_path: Path) -> Dict[str, object]:
    tasks = parse_researcher_key(key_path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)

    best_indices = [i for i, h in enumerate(header) if h.startswith("Q2 (Best):")]
    why_indices = [i for i, h in enumerate(header) if h.startswith("Q3 (Why):")]
    if len(best_indices) != len(tasks):
        raise ValueError(
            f"Task mismatch: survey has {len(best_indices)} task blocks, key has {len(tasks)} tasks"
        )
    if len(why_indices) != len(tasks):
        raise ValueError(
            f"Task mismatch: survey has {len(why_indices)} why blocks, key has {len(tasks)} tasks"
        )

    ranks_by_mode: Dict[str, List[int]] = defaultdict(list)
    firsts_by_mode: Counter[str] = Counter()
    ranks_by_dataset_mode: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    firsts_by_dataset_mode: Counter[Tuple[str, str]] = Counter()
    theme_counts: Counter[str] = Counter()
    theme_counts_by_mode: Dict[str, Counter[str]] = defaultdict(Counter)
    theme_examples: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    respondent_reason_quality: List[Dict[str, object]] = []
    pairwise_wins: Counter[Tuple[str, str]] = Counter()
    pairwise_totals: Counter[Tuple[str, str]] = Counter()
    pairwise_wins_by_dataset: Counter[Tuple[str, str, str]] = Counter()
    pairwise_totals_by_dataset: Counter[Tuple[str, str, str]] = Counter()
    per_task_mode_ranks: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)

    for respondent_idx, row in enumerate(rows, start=1):
        generic_count = 0
        short_count = 0
        reasons_present = 0

        for block_index, best_idx in enumerate(best_indices):
            task = tasks[block_index]
            dataset = normalize_dataset_name(str(task["dataset"]))
            version_to_mode = task["versions"]
            why_idx = why_indices[block_index]

            rank_values = row[best_idx - 5 : best_idx]
            best_choice = row[best_idx].strip()
            why_text = row[why_idx].strip()
            mode_ranks: Dict[str, int] = {}

            for i, version in enumerate(VERSION_LABELS):
                rank = parse_rank(rank_values[i])
                if rank is None:
                    continue
                mode = str(version_to_mode[version])
                ranks_by_mode[mode].append(rank)
                ranks_by_dataset_mode[(dataset, mode)].append(rank)
                mode_ranks[mode] = rank
                per_task_mode_ranks[(str(task["task_id"]), dataset, mode)].append(rank)

            if best_choice in version_to_mode:
                mode = str(version_to_mode[best_choice])
                firsts_by_mode[mode] += 1
                firsts_by_dataset_mode[(dataset, mode)] += 1
                if why_text:
                    reasons_present += 1
                    if len(why_text) < 20:
                        short_count += 1
                    if why_text.lower().strip() in GENERIC_REASON_TEXTS:
                        generic_count += 1
                    themes = classify_reason_themes(why_text)
                    for theme in themes:
                        theme_counts[theme] += 1
                        theme_counts_by_mode[mode][theme] += 1
                        if len(theme_examples[theme]) < 5:
                            theme_examples[theme].append(
                                {
                                    "dataset": dataset,
                                    "mode": mode,
                                    "text": why_text,
                                }
                            )

            for left_mode, right_mode in itertools.combinations(MODE_ORDER, 2):
                if left_mode not in mode_ranks or right_mode not in mode_ranks:
                    continue
                pairwise_totals[(left_mode, right_mode)] += 1
                pairwise_totals_by_dataset[(dataset, left_mode, right_mode)] += 1
                if mode_ranks[left_mode] < mode_ranks[right_mode]:
                    pairwise_wins[(left_mode, right_mode)] += 1
                    pairwise_wins_by_dataset[(dataset, left_mode, right_mode)] += 1
                elif mode_ranks[right_mode] < mode_ranks[left_mode]:
                    pairwise_wins[(right_mode, left_mode)] += 1
                    pairwise_wins_by_dataset[(dataset, right_mode, left_mode)] += 1

        respondent_reason_quality.append(
            {
                "respondent_index": respondent_idx,
                "timestamp": row[0],
                "reasons_present": reasons_present,
                "generic_reason_count": generic_count,
                "short_reason_count_lt20_chars": short_count,
            }
        )

    overall = []
    for mode in sorted(ranks_by_mode.keys(), key=safe_mode_sort_key):
        n = len(ranks_by_mode[mode])
        first_count = int(firsts_by_mode[mode])
        overall.append(
            {
                "mode": mode,
                "n_rankings": n,
                "avg_rank": mean(ranks_by_mode[mode]),
                "first_place_count": first_count,
                "first_place_rate": (first_count / n) if n else 0.0,
            }
        )

    by_dataset = []
    for (dataset, mode), values in sorted(
        ranks_by_dataset_mode.items(),
        key=lambda x: (dataset_sort_key(x[0][0]), safe_mode_sort_key(x[0][1])),
    ):
        n = len(values)
        first_count = int(firsts_by_dataset_mode[(dataset, mode)])
        by_dataset.append(
            {
                "dataset": dataset,
                "mode": mode,
                "n_rankings": n,
                "avg_rank": mean(values),
                "first_place_count": first_count,
                "first_place_rate": (first_count / n) if n else 0.0,
            }
        )

    pairwise_overall: List[Dict[str, object]] = []
    pairwise_by_dataset: List[Dict[str, object]] = []
    dataset_ids = sorted({ds for ds, _ in ranks_by_dataset_mode.keys()}, key=dataset_sort_key)
    for left_mode, right_mode in itertools.combinations(MODE_ORDER, 2):
        total = int(pairwise_totals[(left_mode, right_mode)])
        if total == 0:
            continue
        left_wins = int(pairwise_wins[(left_mode, right_mode)])
        right_wins = int(pairwise_wins[(right_mode, left_mode)])
        pairwise_overall.append(
            {
                "left_mode": left_mode,
                "right_mode": right_mode,
                "comparisons": total,
                "left_win_rate": left_wins / total,
                "right_win_rate": right_wins / total,
            }
        )
        for dataset in dataset_ids:
            ds_total = int(pairwise_totals_by_dataset[(dataset, left_mode, right_mode)])
            if ds_total == 0:
                continue
            ds_left_wins = int(pairwise_wins_by_dataset[(dataset, left_mode, right_mode)])
            ds_right_wins = int(pairwise_wins_by_dataset[(dataset, right_mode, left_mode)])
            pairwise_by_dataset.append(
                {
                    "dataset": dataset,
                    "left_mode": left_mode,
                    "right_mode": right_mode,
                    "comparisons": ds_total,
                    "left_win_rate": ds_left_wins / ds_total,
                    "right_win_rate": ds_right_wins / ds_total,
                }
            )

    per_task_table: List[Dict[str, object]] = []
    task_winner_counts_overall: Dict[str, float] = defaultdict(float)
    task_winner_counts_by_dataset: Dict[Tuple[str, str], float] = defaultdict(float)
    task_groups: Dict[Tuple[str, str], Dict[str, List[int]]] = defaultdict(dict)
    for (task_id, dataset, mode), values in per_task_mode_ranks.items():
        task_groups[(task_id, dataset)][mode] = values
    for (task_id, dataset), mode_map in sorted(task_groups.items(), key=lambda x: x[0][0]):
        avg_by_mode: Dict[str, float] = {}
        for mode, values in mode_map.items():
            if values:
                avg_by_mode[mode] = mean(values)
        if not avg_by_mode:
            continue
        best_rank = min(avg_by_mode.values())
        winners = [m for m, r in avg_by_mode.items() if r == best_rank]
        share = 1.0 / len(winners)
        for winner in winners:
            task_winner_counts_overall[winner] += share
            task_winner_counts_by_dataset[(dataset, winner)] += share
        per_task_table.append(
            {
                "task_id": task_id,
                "dataset": dataset,
                "winner_modes": ",".join(sorted(winners, key=safe_mode_sort_key)),
                "winning_avg_rank": best_rank,
                "avg_rank_original": avg_by_mode.get("original"),
                "avg_rank_generated_raw": avg_by_mode.get("generated_raw"),
                "avg_rank_generated_cmg_only": avg_by_mode.get("generated_cmg_only"),
                "avg_rank_generated_file_summaries_only": avg_by_mode.get(
                    "generated_file_summaries_only"
                ),
                "avg_rank_generated_full": avg_by_mode.get("generated_full"),
            }
        )

    task_winner_summary_overall: List[Dict[str, object]] = []
    task_count = len(per_task_table)
    for mode in MODE_ORDER:
        wins = float(task_winner_counts_overall.get(mode, 0.0))
        task_winner_summary_overall.append(
            {
                "mode": mode,
                "task_wins": wins,
                "task_win_rate": (wins / task_count) if task_count else 0.0,
            }
        )
    task_winner_summary_by_dataset: List[Dict[str, object]] = []
    for dataset in dataset_ids:
        ds_task_count = sum(1 for rec in per_task_table if rec["dataset"] == dataset)
        for mode in MODE_ORDER:
            wins = float(task_winner_counts_by_dataset.get((dataset, mode), 0.0))
            task_winner_summary_by_dataset.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "task_wins": wins,
                    "task_win_rate": (wins / ds_task_count) if ds_task_count else 0.0,
                }
            )

    payload: Dict[str, object] = {
        "inputs": {
            "survey_csv": str(csv_path.name),
            "researcher_key": str(key_path.name),
        },
        "meta": {
            "respondents": len(rows),
            "tasks_per_respondent": len(best_indices),
            "total_rankings": len(rows) * len(best_indices) * 5,
            "total_first_place_votes": len(rows) * len(best_indices),
        },
        "overall": overall,
        "by_dataset": by_dataset,
        "pairwise_preference": {
            "overall": pairwise_overall,
            "by_dataset": pairwise_by_dataset,
        },
        "task_winner_analysis": {
            "per_task": per_task_table,
            "overall": task_winner_summary_overall,
            "by_dataset": task_winner_summary_by_dataset,
        },
        "qualitative": {
            "theme_counts_overall": dict(theme_counts.most_common()),
            "theme_counts_by_mode": {
                mode: dict(counter.most_common())
                for mode, counter in sorted(
                    theme_counts_by_mode.items(), key=lambda x: safe_mode_sort_key(x[0])
                )
            },
            "theme_examples": dict(theme_examples),
            "respondent_reason_quality": respondent_reason_quality,
        },
    }

    return payload


def write_csv(path: Path, payload: Dict[str, object]) -> None:
    rows: List[Dict[str, object]] = []
    for rec in payload["overall"]:
        rows.append({"scope": "overall", "dataset": "all", **rec})
    for rec in payload["by_dataset"]:
        rows.append({"scope": "dataset", **rec, "dataset": label_dataset(str(rec["dataset"]))})

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scope",
                "dataset",
                "mode",
                "n_rankings",
                "avg_rank",
                "first_place_count",
                "first_place_rate",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_markdown_table(rows: List[Dict[str, object]], columns: List[str]) -> str:
    out = []
    out.append("| " + " | ".join(columns) + " |")
    out.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        vals: List[str] = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def find_metric(rows: List[Dict[str, object]], dataset: str, mode: str, key: str) -> float:
    for rec in rows:
        if rec.get("dataset") == dataset and rec.get("mode") == mode:
            return float(rec.get(key, 0.0))
    return 0.0


def best_mode(rows: List[Dict[str, object]], dataset: str, key: str, higher_better: bool) -> str:
    filtered = [r for r in rows if r.get("dataset") == dataset]
    if not filtered:
        return "n/a"
    ordered = sorted(filtered, key=lambda r: float(r.get(key, 0.0)), reverse=higher_better)
    return str(ordered[0].get("mode"))


def write_md(path: Path, payload: Dict[str, object]) -> None:
    meta = payload["meta"]
    by_dataset_rows = payload["by_dataset"]
    task_winner_rows = payload["task_winner_analysis"]["by_dataset"]
    original_first_aidev = find_metric(by_dataset_rows, "aidev", "original", "first_place_rate")
    original_first_trudeau = find_metric(by_dataset_rows, "prsummarizer", "original", "first_place_rate")
    best_aidev_avg_rank = best_mode(by_dataset_rows, "aidev", "avg_rank", higher_better=False)
    best_trudeau_avg_rank = best_mode(by_dataset_rows, "prsummarizer", "avg_rank", higher_better=False)
    best_aidev_task = best_mode(task_winner_rows, "aidev", "task_win_rate", higher_better=True)
    best_trudeau_task = best_mode(task_winner_rows, "prsummarizer", "task_win_rate", higher_better=True)

    lines: List[str] = []
    lines.append("# Human Evaluation Summary")
    lines.append("")
    lines.append("## Key Findings")
    lines.append(
        f"- Original was still selected first in both datasets: AIDev `{pct(original_first_aidev)}` and PRSummarizer `{pct(original_first_trudeau)}`."
    )
    lines.append(
        f"- Best mode by average rank (lower is better): AIDev `{best_aidev_avg_rank}`; PRSummarizer `{best_trudeau_avg_rank}`."
    )
    lines.append(
        f"- Best mode by per-task winner rate: AIDev `{best_aidev_task}`; PRSummarizer `{best_trudeau_task}`."
    )
    lines.append("")
    lines.append("## Meta")
    lines.append(f"- Respondents: {meta['respondents']}")
    lines.append(f"- Tasks per respondent: {meta['tasks_per_respondent']}")
    lines.append(f"- Total rankings: {meta['total_rankings']}")
    lines.append(
        f"- Total first-place votes: {meta['total_first_place_votes']} (10 respondents x 20 tasks)"
    )
    lines.append("")
    lines.append("## Overall")
    lines.append(
        to_markdown_table(
            payload["overall"],
            ["mode", "n_rankings", "avg_rank", "first_place_count", "first_place_rate"],
        )
    )
    lines.append("")
    lines.append("## By Dataset")
    by_dataset_labeled = [{**r, "dataset": label_dataset(str(r["dataset"]))} for r in payload["by_dataset"]]
    lines.append(
        to_markdown_table(
            by_dataset_labeled,
            ["dataset", "mode", "n_rankings", "avg_rank", "first_place_count", "first_place_rate"],
        )
    )
    lines.append("")
    lines.append("## Pairwise Preference")
    lines.append(
        to_markdown_table(
            payload["pairwise_preference"]["overall"],
            ["left_mode", "right_mode", "comparisons", "left_win_rate", "right_win_rate"],
        )
    )
    lines.append("")
    lines.append("## Per-Task Winner Rate")
    lines.append(
        to_markdown_table(
            payload["task_winner_analysis"]["overall"],
            ["mode", "task_wins", "task_win_rate"],
        )
    )
    lines.append("")
    lines.append("## Per-Task Winner Rate By Dataset")
    task_winner_labeled = [
        {**r, "dataset": label_dataset(str(r["dataset"]))}
        for r in payload["task_winner_analysis"]["by_dataset"]
    ]
    lines.append(
        to_markdown_table(
            task_winner_labeled,
            ["dataset", "mode", "task_wins", "task_win_rate"],
        )
    )
    lines.append("")
    lines.append("## Why Text Themes")
    theme_rows = [
        {"theme": theme, "count": count}
        for theme, count in payload["qualitative"]["theme_counts_overall"].items()
    ]
    lines.append(to_markdown_table(theme_rows, ["theme", "count"]))
    lines.append("")
    lines.append("## Response Quality (Per Respondent)")
    lines.append(
        to_markdown_table(
            payload["qualitative"]["respondent_reason_quality"],
            [
                "respondent_index",
                "timestamp",
                "reasons_present",
                "generic_reason_count",
                "short_reason_count_lt20_chars",
            ],
        )
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_human_plot(path: Path, payload: Dict[str, object]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode_order = [
        "original",
        "generated_raw",
        "generated_cmg_only",
        "generated_file_summaries_only",
        "generated_full",
    ]
    labels = {
        "original": "Original",
        "generated_raw": "Raw",
        "generated_cmg_only": "CMG",
        "generated_file_summaries_only": "FileSum",
        "generated_full": "Full",
    }
    colors = {"aidev": "#1f77b4", "prsummarizer": "#ff7f0e"}

    by_dataset = payload.get("by_dataset", [])
    task_winners = payload.get("task_winner_analysis", {}).get("by_dataset", [])

    idx_rank: Dict[Tuple[str, str], float] = {}
    idx_first: Dict[Tuple[str, str], float] = {}
    idx_task: Dict[Tuple[str, str], float] = {}
    for rec in by_dataset:
        idx_rank[(rec["dataset"], rec["mode"])] = float(rec["avg_rank"])
        idx_first[(rec["dataset"], rec["mode"])] = float(rec["first_place_rate"])
    for rec in task_winners:
        idx_task[(rec["dataset"], rec["mode"])] = float(rec["task_win_rate"])
    dataset_keys = sorted({str(rec["dataset"]) for rec in by_dataset}, key=dataset_sort_key)

    metrics = [
        ("avg_rank", "Human Avg Rank (lower better)", idx_rank, 0.0, 5.0),
        ("first_place_rate", "Human First-Place Rate", idx_first, 0.0, 0.4),
        ("task_win_rate", "Per-Task Winner Rate", idx_task, 0.0, 0.8),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.0), constrained_layout=False)
    fig.suptitle("Human Evaluation by Mode", fontsize=14, y=0.97)

    x = list(range(len(mode_order)))
    n_ds = max(1, len(dataset_keys))
    width = 0.36 if n_ds == 2 else 0.6

    for ax, (_key, title, idx, y_min, y_max) in zip(axes, metrics):
        for ds_idx, ds in enumerate(dataset_keys):
            offset = (ds_idx - (n_ds - 1) / 2) * width
            vals = [float(idx.get((ds, mode), 0.0)) for mode in mode_order]
            ax.bar([i + offset for i in x], vals, width=width, color=colors[ds], label=label_dataset(ds))
        ax.set_ylim(y_min, y_max)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in mode_order], rotation=15, ha="right", fontsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0.03)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=max(1, len(legend_labels)),
        bbox_to_anchor=(0.5, 0.91),
        fontsize=10,
        frameon=False,
    )
    fig.subplots_adjust(left=0.05, right=0.995, top=0.82, bottom=0.19, wspace=0.22)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def sync_results_to_research_paper() -> Path | None:
    if not RESEARCH_PAPER_DIR.exists():
        return None

    PAPER_FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plots_src = ROOT / "eval" / "plots"
    tables_src = ROOT / "eval" / "tables"
    plots_dst = PAPER_FINAL_RESULTS_DIR / "plots"
    tables_dst = PAPER_FINAL_RESULTS_DIR / "tables"
    if plots_src.exists():
        shutil.copytree(plots_src, plots_dst, dirs_exist_ok=True)
    if tables_src.exists():
        shutil.copytree(tables_src, tables_dst, dirs_exist_ok=True)
    return PAPER_FINAL_RESULTS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze human evaluation survey results")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Survey CSV path")
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY, help="Researcher key markdown path")
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Output file prefix name (without extension)",
    )
    args = parser.parse_args()

    payload = analyze(args.csv, args.key)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = TABLES_DIR / f"{args.out_prefix}.json"
    csv_path = TABLES_DIR / f"{args.out_prefix}.csv"
    md_path = TABLES_DIR / f"{args.out_prefix}.md"
    png_path = PLOTS_DIR / f"{args.out_prefix}.png"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, payload)
    write_md(md_path, payload)
    written_plot = write_human_plot(png_path, payload)
    synced_dir = sync_results_to_research_paper()

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {written_plot}")
    if synced_dir is not None:
        print(f"Synced final-results to {synced_dir}")
    else:
        print("Skipped sync: research-paper directory not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
