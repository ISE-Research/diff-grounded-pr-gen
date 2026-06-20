#!/usr/bin/env python3
"""Analyze agreement between human evaluation rankings and LLM-as-judge scores.

Inputs:
- data/human-data/Human Evaluation Survey- Pull Request Descriptions.csv
- data/human-data/survey_researcher_key.md
- data/description-data/*-judge-*.json

Outputs:
- eval/tables/human_llm_agreement_summary.json
- eval/tables/human_llm_agreement_summary.csv
- eval/tables/human_llm_agreement_summary.md
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HUMAN_DATA_DIR = ROOT / "data" / "human-data"
DESCRIPTION_DATA_DIR = ROOT / "data" / "description-data"
TABLES_DIR = ROOT / "eval" / "tables"
DEFAULT_CSV = HUMAN_DATA_DIR / "Human Evaluation Survey- Pull Request Descriptions.csv"
DEFAULT_KEY = HUMAN_DATA_DIR / "survey_researcher_key.md"
DEFAULT_JUDGE_SUMMARY = TABLES_DIR / "descriptions_summary.csv"
DEFAULT_PREFIX = "human_llm_agreement_summary"

VERSION_LABELS = ("A", "B", "C", "D", "E")
MODE_ORDER = (
    "original",
    "generated_raw",
    "generated_cmg_only",
    "generated_file_summaries_only",
    "generated_full",
)
JUDGE_MODE_MAP = {
    "raw": "generated_raw",
    "cmg_only": "generated_cmg_only",
    "file_summaries_only": "generated_file_summaries_only",
    "full": "generated_full",
}
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


def parse_rank(text: str) -> int | None:
    if not text:
        return None
    match = re.match(r"\s*([1-5])", text)
    if not match:
        return None
    return int(match.group(1))


def normalize_dataset_name(dataset: str) -> str:
    return DATASET_ALIASES.get(dataset.lower(), dataset.lower())


def dataset_sort_key(dataset: str) -> Tuple[int, str]:
    return (DATASET_ORDER.get(dataset, 99), dataset)


def mode_sort_key(mode: str) -> Tuple[int, str]:
    if mode in MODE_ORDER:
        return (MODE_ORDER.index(mode), mode)
    return (len(MODE_ORDER), mode)


def label_dataset(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def parse_researcher_key(path: Path) -> List[Dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    tasks: List[Dict[str, object]] = []
    current: Dict[str, object] | None = None

    header_re = re.compile(
        r"##\s+([PA]\d+)\s+\((parsed|trudeau|aidev)\)\s+.\s+(.+?)\s+PR\s+#(\d+)"
    )
    version_re = re.compile(r"- Version ([A-E]):\s+([a-z_]+)")

    for line in lines:
        header = header_re.match(line.strip())
        if header:
            current = {
                "task_id": header.group(1),
                "dataset": normalize_dataset_name(header.group(2)),
                "repo_name": header.group(3).strip(),
                "pr_number": int(header.group(4)),
                "versions": {},
            }
            tasks.append(current)
            continue

        version = version_re.match(line.strip())
        if version and current is not None:
            current["versions"][version.group(1)] = version.group(2)

    for task in tasks:
        versions = task["versions"]
        missing = [v for v in VERSION_LABELS if v not in versions]
        if missing:
            raise ValueError(f"Missing version mappings for {task['task_id']}: {missing}")

    return tasks


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def average_rank(values_by_mode: Dict[str, float], lower_is_better: bool) -> Dict[str, float]:
    sorted_items = sorted(
        values_by_mode.items(),
        key=lambda item: (item[1] if lower_is_better else -item[1], mode_sort_key(item[0])),
    )
    ranks: Dict[str, float] = {}
    pos = 1
    for _, group in itertools.groupby(sorted_items, key=lambda item: item[1]):
        tied = list(group)
        avg = mean(range(pos, pos + len(tied)))
        for mode, _ in tied:
            ranks[mode] = avg
        pos += len(tied)
    return ranks


def best_modes(values_by_mode: Dict[str, float], lower_is_better: bool) -> List[str]:
    if not values_by_mode:
        return []
    best_value = min(values_by_mode.values()) if lower_is_better else max(values_by_mode.values())
    return sorted(
        [mode for mode, value in values_by_mode.items() if value == best_value],
        key=mode_sort_key,
    )


def load_human_tasks(csv_path: Path, key_path: Path) -> List[Dict[str, object]]:
    tasks = parse_researcher_key(key_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)

    best_indices = [i for i, h in enumerate(header) if h.startswith("Q2 (Best):")]
    if len(best_indices) != len(tasks):
        raise ValueError(
            f"Task mismatch: survey has {len(best_indices)} task blocks, key has {len(tasks)} tasks"
        )

    output: List[Dict[str, object]] = []
    for block_index, best_idx in enumerate(best_indices):
        task = tasks[block_index]
        version_to_mode = task["versions"]
        ranks_by_mode: Dict[str, List[int]] = defaultdict(list)
        first_choice_counts: Counter[str] = Counter()

        for row in rows:
            rank_values = row[best_idx - 5 : best_idx]
            for i, version in enumerate(VERSION_LABELS):
                rank = parse_rank(rank_values[i])
                if rank is None:
                    continue
                mode = str(version_to_mode[version])
                ranks_by_mode[mode].append(rank)

            best_choice = row[best_idx].strip()
            if best_choice in version_to_mode:
                first_choice_counts[str(version_to_mode[best_choice])] += 1

        avg_ranks = {mode: mean(values) for mode, values in ranks_by_mode.items()}
        output.append(
            {
                **task,
                "human_avg_rank_by_mode": avg_ranks,
                "human_rank_position_by_mode": average_rank(avg_ranks, lower_is_better=True),
                "human_winner_modes": best_modes(avg_ranks, lower_is_better=True),
                "human_first_choice_counts": dict(first_choice_counts),
            }
        )

    return output


def judge_dataset_from_path(path: Path) -> str:
    name = path.name.lower()
    if "aidev" in name:
        return "aidev"
    if "parsed" in name or "trudeau" in name:
        return "prsummarizer"
    return "unknown"


def load_llm_scores(paths: List[Path]) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    by_task: Dict[Tuple[str, str, int], Dict[str, float]] = defaultdict(dict)
    for path in paths:
        dataset = judge_dataset_from_path(path)
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"Expected list in {path}")

        for row in rows:
            repo_name = row.get("repo_name")
            pr_number = row.get("pr_number")
            mode = JUDGE_MODE_MAP.get(str(row.get("generation_mode", "")).strip().lower())
            judgment = row.get("judgment") or {}
            if not repo_name or pr_number is None or not mode:
                continue
            key = (dataset, str(repo_name), int(pr_number))
            by_task[key]["original"] = float(judgment.get("original_score"))
            by_task[key][mode] = float(judgment.get("generated_score"))
    return by_task


def load_judge_summary(path: Path) -> Dict[str, Dict[str, float]]:
    by_dataset: Dict[str, Dict[str, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = normalize_dataset_name(str(row["dataset"]))
            mode = str(row["mode"])
            if mode in JUDGE_MODE_MAP:
                mode = JUDGE_MODE_MAP[mode]
            if mode not in MODE_ORDER:
                continue
            by_dataset[dataset][mode] = float(row["avg_generated_score"])
    return by_dataset


def human_aggregate_ranks(
    human_tasks: List[Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for task in human_tasks:
        dataset = str(task["dataset"])
        avg_ranks = dict(task["human_avg_rank_by_mode"])
        for mode, rank in avg_ranks.items():
            grouped["all"][mode].append(float(rank))
            grouped[dataset][mode].append(float(rank))
    return {
        dataset: {mode: mean(values) for mode, values in mode_map.items()}
        for dataset, mode_map in grouped.items()
    }


def judge_aggregate_scores(
    judge_summary: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for dataset, mode_map in judge_summary.items():
        for mode, score in mode_map.items():
            grouped["all"][mode].append(float(score))
            grouped[dataset][mode].append(float(score))
    return {
        dataset: {mode: mean(values) for mode, values in mode_map.items()}
        for dataset, mode_map in grouped.items()
    }


def aggregate_alignment(
    human_tasks: List[Dict[str, object]],
    judge_summary_path: Path,
) -> List[Dict[str, object]]:
    human_by_scope = human_aggregate_ranks(human_tasks)
    judge_by_scope = judge_aggregate_scores(load_judge_summary(judge_summary_path))
    scopes = ["all"] + sorted(
        (set(human_by_scope.keys()) & set(judge_by_scope.keys())) - {"all"},
        key=dataset_sort_key,
    )

    rows: List[Dict[str, object]] = []
    generated_modes = [mode for mode in MODE_ORDER if mode != "original"]
    for dataset in scopes:
        human_values = human_by_scope.get(dataset, {})
        judge_values = judge_by_scope.get(dataset, {})
        if any(mode not in human_values or mode not in judge_values for mode in MODE_ORDER):
            continue

        human_positions = average_rank(human_values, lower_is_better=True)
        judge_positions = average_rank(judge_values, lower_is_better=False)
        tau = kendall_tau_b(human_positions, judge_positions, MODE_ORDER)
        human_winners = best_modes(human_values, lower_is_better=True)
        judge_winners = best_modes(judge_values, lower_is_better=False)

        generated_beats_original = []
        for mode in generated_modes:
            human_prefers_generated = human_values[mode] < human_values["original"]
            judge_prefers_generated = judge_values[mode] > judge_values["original"]
            generated_beats_original.append(human_prefers_generated == judge_prefers_generated)

        rows.append(
            {
                "scope": "overall" if dataset == "all" else "dataset",
                "dataset": dataset,
                "human_top_modes": ",".join(human_winners),
                "llm_top_modes": ",".join(judge_winners),
                "top_mode_overlap": bool(set(human_winners) & set(judge_winners)),
                "aggregate_kendall_tau_b": tau["tau_b"],
                "generated_vs_original_direction_agreement": mean(
                    1.0 if agrees else 0.0 for agrees in generated_beats_original
                ),
                "human_original_avg_rank": human_values["original"],
                "llm_original_avg_score": judge_values["original"],
                "human_mode_order": ",".join(
                    mode for mode, _ in sorted(human_values.items(), key=lambda item: (item[1], mode_sort_key(item[0])))
                ),
                "llm_mode_order": ",".join(
                    mode for mode, _ in sorted(judge_values.items(), key=lambda item: (-item[1], mode_sort_key(item[0])))
                ),
                **tau,
            }
        )
    return rows


def pairwise_direction_rows(
    human_tasks: List[Dict[str, object]],
    judge_summary_path: Path,
) -> List[Dict[str, object]]:
    human_by_scope = human_aggregate_ranks(human_tasks)
    judge_by_scope = judge_aggregate_scores(load_judge_summary(judge_summary_path))
    scopes = ["all"] + sorted(
        (set(human_by_scope.keys()) & set(judge_by_scope.keys())) - {"all"},
        key=dataset_sort_key,
    )
    comparisons = [
        ("original", mode)
        for mode in MODE_ORDER
        if mode != "original"
    ] + [
        ("generated_raw", mode)
        for mode in MODE_ORDER
        if mode not in {"original", "generated_raw"}
    ]

    rows: List[Dict[str, object]] = []
    for dataset in scopes:
        human_values = human_by_scope.get(dataset, {})
        judge_values = judge_by_scope.get(dataset, {})
        if any(mode not in human_values or mode not in judge_values for mode in MODE_ORDER):
            continue
        for baseline, mode in comparisons:
            human_delta_rank = human_values[baseline] - human_values[mode]
            judge_delta_score = judge_values[mode] - judge_values[baseline]
            human_prefers_mode = human_delta_rank > 0
            judge_prefers_mode = judge_delta_score > 0
            rows.append(
                {
                    "scope": "overall" if dataset == "all" else "dataset",
                    "dataset": dataset,
                    "baseline": baseline,
                    "mode": mode,
                    "human_delta_rank_improvement": human_delta_rank,
                    "llm_delta_score_improvement": judge_delta_score,
                    "human_prefers_mode_over_baseline": human_prefers_mode,
                    "llm_prefers_mode_over_baseline": judge_prefers_mode,
                    "direction_agrees": human_prefers_mode == judge_prefers_mode,
                }
            )
    return rows


def cohen_kappa(pairs: List[Tuple[str, str]], labels: Tuple[str, ...]) -> Dict[str, object]:
    n = len(pairs)
    if n == 0:
        return {"n": 0, "observed_agreement": 0.0, "expected_agreement": 0.0, "kappa": 0.0}

    observed = sum(1 for left, right in pairs if left == right) / n
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum((left_counts[label] / n) * (right_counts[label] / n) for label in labels)
    kappa = (observed - expected) / (1 - expected) if expected != 1 else 0.0
    return {
        "n": n,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
    }


def kendall_tau_b(left: Dict[str, float], right: Dict[str, float], labels: Tuple[str, ...]) -> Dict[str, object]:
    concordant = 0
    discordant = 0
    left_ties = 0
    right_ties = 0
    both_ties = 0

    for a, b in itertools.combinations(labels, 2):
        left_diff = left[a] - left[b]
        right_diff = right[a] - right[b]
        if left_diff == 0 and right_diff == 0:
            both_ties += 1
        elif left_diff == 0:
            left_ties += 1
        elif right_diff == 0:
            right_ties += 1
        elif left_diff * right_diff > 0:
            concordant += 1
        else:
            discordant += 1

    denominator = math.sqrt(
        (concordant + discordant + left_ties)
        * (concordant + discordant + right_ties)
    )
    tau = (concordant - discordant) / denominator if denominator else 0.0
    return {
        "tau_b": tau,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "human_tied_pairs": left_ties,
        "llm_tied_pairs": right_ties,
        "both_tied_pairs": both_ties,
    }


def summarize_scope(rows: List[Dict[str, object]], scope: str, dataset: str) -> Dict[str, object]:
    kappa_pairs = [
        (str(row["human_winner_mode_for_kappa"]), str(row["llm_winner_mode_for_kappa"]))
        for row in rows
        if row.get("human_winner_mode_for_kappa") and row.get("llm_winner_mode_for_kappa")
    ]
    taus = [float(row["kendall_tau_b"]) for row in rows if row.get("kendall_tau_b") is not None]
    kappa = cohen_kappa(kappa_pairs, MODE_ORDER)
    return {
        "scope": scope,
        "dataset": dataset,
        "tasks": len(rows),
        "cohen_kappa": kappa["kappa"],
        "observed_winner_agreement": kappa["observed_agreement"],
        "expected_winner_agreement": kappa["expected_agreement"],
        "mean_kendall_tau_b": mean(taus),
        "median_kendall_tau_b": median(taus),
    }


def analyze(
    csv_path: Path,
    key_path: Path,
    judge_paths: List[Path],
    judge_summary_path: Path,
) -> Dict[str, object]:
    human_tasks = load_human_tasks(csv_path, key_path)
    llm_scores = load_llm_scores(judge_paths)
    aggregate_rows = aggregate_alignment(human_tasks, judge_summary_path)
    pairwise_rows = pairwise_direction_rows(human_tasks, judge_summary_path)

    per_task: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    for task in human_tasks:
        key = (str(task["dataset"]), str(task["repo_name"]), int(task["pr_number"]))
        scores = llm_scores.get(key)
        if not scores or any(mode not in scores for mode in MODE_ORDER):
            missing.append(
                {
                    "task_id": task["task_id"],
                    "repo_name": task["repo_name"],
                    "pr_number": task["pr_number"],
                }
            )
            continue

        human_avg_ranks = dict(task["human_avg_rank_by_mode"])
        human_rank_positions = dict(task["human_rank_position_by_mode"])
        llm_rank_positions = average_rank(scores, lower_is_better=False)
        human_winners = list(task["human_winner_modes"])
        llm_winners = best_modes(scores, lower_is_better=False)

        # Kappa is defined on one category per rater. If a task ties, use a stable
        # mode-order tie break and preserve the tie details in the per-task table.
        human_winner = human_winners[0] if human_winners else None
        llm_winner = llm_winners[0] if llm_winners else None
        tau = kendall_tau_b(human_rank_positions, llm_rank_positions, MODE_ORDER)

        per_task.append(
            {
                "task_id": task["task_id"],
                "dataset": task["dataset"],
                "repo_name": task["repo_name"],
                "pr_number": task["pr_number"],
                "human_winner_modes": ",".join(human_winners),
                "llm_winner_modes": ",".join(llm_winners),
                "human_winner_mode_for_kappa": human_winner,
                "llm_winner_mode_for_kappa": llm_winner,
                "winner_agrees": human_winner == llm_winner,
                "kendall_tau_b": tau["tau_b"],
                "human_avg_rank_by_mode": human_avg_ranks,
                "llm_score_by_mode": scores,
                "human_rank_position_by_mode": human_rank_positions,
                "llm_rank_position_by_mode": llm_rank_positions,
                **tau,
            }
        )

    summaries = [summarize_scope(per_task, "overall", "all")]
    for dataset in sorted({str(row["dataset"]) for row in per_task}, key=dataset_sort_key):
        dataset_rows = [row for row in per_task if row["dataset"] == dataset]
        summaries.append(summarize_scope(dataset_rows, "dataset", dataset))

    return {
        "inputs": {
            "survey_csv": str(csv_path),
            "researcher_key": str(key_path),
            "judge_files": [str(path) for path in judge_paths],
            "judge_summary_csv": str(judge_summary_path),
        },
        "meta": {
            "human_tasks": len(human_tasks),
            "aligned_tasks": len(per_task),
            "missing_llm_tasks": len(missing),
        },
        "aggregate_alignment": aggregate_rows,
        "pairwise_direction_alignment": pairwise_rows,
        "summary": summaries,
        "per_task": per_task,
        "missing": missing,
    }


def write_csv(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "dataset",
        "tasks",
        "cohen_kappa",
        "observed_winner_agreement",
        "expected_winner_agreement",
        "mean_kendall_tau_b",
        "median_kendall_tau_b",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["summary"]:
            out = dict(row)
            if out["dataset"] != "all":
                out["dataset"] = label_dataset(str(out["dataset"]))
            writer.writerow(out)


def write_aggregate_csv(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "dataset",
        "human_top_modes",
        "llm_top_modes",
        "top_mode_overlap",
        "aggregate_kendall_tau_b",
        "generated_vs_original_direction_agreement",
        "human_original_avg_rank",
        "llm_original_avg_score",
        "human_mode_order",
        "llm_mode_order",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["aggregate_alignment"]:
            out = {key: row[key] for key in fieldnames}
            if out["dataset"] != "all":
                out["dataset"] = label_dataset(str(out["dataset"]))
            writer.writerow(out)


def write_pairwise_csv(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "dataset",
        "baseline",
        "mode",
        "human_delta_rank_improvement",
        "llm_delta_score_improvement",
        "human_prefers_mode_over_baseline",
        "llm_prefers_mode_over_baseline",
        "direction_agrees",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["pairwise_direction_alignment"]:
            out = {key: row[key] for key in fieldnames}
            if out["dataset"] != "all":
                out["dataset"] = label_dataset(str(out["dataset"]))
            writer.writerow(out)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: List[Dict[str, object]], columns: List[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_md(path: Path, payload: Dict[str, object]) -> None:
    aggregate_rows = []
    for row in payload["aggregate_alignment"]:
        aggregate_rows.append(
            {
                "scope": row["scope"],
                "dataset": "all" if row["dataset"] == "all" else label_dataset(str(row["dataset"])),
                "human_top_modes": row["human_top_modes"],
                "llm_top_modes": row["llm_top_modes"],
                "top_mode_overlap": row["top_mode_overlap"],
                "aggregate_kendall_tau_b": row["aggregate_kendall_tau_b"],
                "generated_vs_original_direction_agreement": row[
                    "generated_vs_original_direction_agreement"
                ],
            }
        )

    pairwise_rows = []
    for row in payload["pairwise_direction_alignment"]:
        pairwise_rows.append(
            {
                "scope": row["scope"],
                "dataset": "all" if row["dataset"] == "all" else label_dataset(str(row["dataset"])),
                "baseline": row["baseline"],
                "mode": row["mode"],
                "human_delta_rank_improvement": row["human_delta_rank_improvement"],
                "llm_delta_score_improvement": row["llm_delta_score_improvement"],
                "direction_agrees": row["direction_agrees"],
            }
        )

    summary_rows = []
    for row in payload["summary"]:
        out = dict(row)
        if out["dataset"] != "all":
            out["dataset"] = label_dataset(str(out["dataset"]))
        summary_rows.append(out)

    per_task_rows = []
    for row in payload["per_task"]:
        per_task_rows.append(
            {
                "task_id": row["task_id"],
                "dataset": label_dataset(str(row["dataset"])),
                "human_winner": row["human_winner_modes"],
                "llm_winner": row["llm_winner_modes"],
                "winner_agrees": row["winner_agrees"],
                "kendall_tau_b": row["kendall_tau_b"],
            }
        )

    lines = [
        "# Human vs. LLM-as-Judge Agreement",
        "",
        "## Interpretation",
        "- Aggregate alignment is the appropriate metric for the paper's main claim: whether human rankings and LLM-as-judge scores support the same broad conclusions across modes.",
        "- Pairwise direction alignment compares each generated mode against the original baseline and each enriched mode against the raw zero-shot baseline. Positive human delta means lower/better average rank than the baseline; positive LLM delta means higher judge score than the baseline.",
        "- Matched-task agreement is stricter: it checks whether both evaluators choose/order the same mode for the same PR. It is reported as a diagnostic only because only a subset of human-study PRs has matching judge artifacts.",
        "- Cohen's kappa compares the per-task winning mode chosen by humans against the per-task winning mode from LLM judge scores. Higher is better; 0 means chance-level agreement, negative means worse than chance.",
        "- Kendall's tau-b compares the full five-mode ranking for each task. Human rankings use mean respondent rank; LLM rankings use judge scores. Higher is better; 1 is identical ordering, 0 is no ordinal association, and negative means inverse ordering.",
        "- Lower human rank is better; higher LLM score is better.",
        "- Kappa requires one label per task, so tied winners are resolved with the fixed mode order while the full tie set remains visible in the per-task table.",
        "",
        "## Aggregate Alignment",
        markdown_table(
            aggregate_rows,
            [
                "scope",
                "dataset",
                "human_top_modes",
                "llm_top_modes",
                "top_mode_overlap",
                "aggregate_kendall_tau_b",
                "generated_vs_original_direction_agreement",
            ],
        ),
        "",
        "## Pairwise Direction Alignment",
        markdown_table(
            pairwise_rows,
            [
                "scope",
                "dataset",
                "baseline",
                "mode",
                "human_delta_rank_improvement",
                "llm_delta_score_improvement",
                "direction_agrees",
            ],
        ),
        "",
        "## Matched-Task Diagnostic Summary",
        markdown_table(
            summary_rows,
            [
                "scope",
                "dataset",
                "tasks",
                "cohen_kappa",
                "observed_winner_agreement",
                "expected_winner_agreement",
                "mean_kendall_tau_b",
                "median_kendall_tau_b",
            ],
        ),
        "",
        "## Per Task",
        markdown_table(
            per_task_rows,
            [
                "task_id",
                "dataset",
                "human_winner",
                "llm_winner",
                "winner_agrees",
                "kendall_tau_b",
            ],
        ),
    ]
    if payload["missing"]:
        lines.extend(["", "## Missing LLM Tasks", markdown_table(payload["missing"], ["task_id", "repo_name", "pr_number"])])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument(
        "--judge",
        type=Path,
        action="append",
        default=None,
        help="Judge JSON path. Defaults to the two full judge artifacts in data/description-data.",
    )
    parser.add_argument("--judge-summary", type=Path, default=DEFAULT_JUDGE_SUMMARY)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    judge_paths = args.judge
    if judge_paths is None:
        judge_paths = sorted(DESCRIPTION_DATA_DIR.glob("*.json"))
    payload = analyze(args.csv, args.key, judge_paths, args.judge_summary)

    output_json = TABLES_DIR / f"{args.prefix}.json"
    output_csv = TABLES_DIR / f"{args.prefix}.csv"
    output_aggregate_csv = TABLES_DIR / f"{args.prefix}_aggregate.csv"
    output_pairwise_csv = TABLES_DIR / f"{args.prefix}_pairwise.csv"
    output_md = TABLES_DIR / f"{args.prefix}.md"

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(output_csv, payload)
    write_aggregate_csv(output_aggregate_csv, payload)
    write_pairwise_csv(output_pairwise_csv, payload)
    write_md(output_md, payload)

    print(f"[AGREEMENT] Wrote {output_json}")
    print(f"[AGREEMENT] Wrote {output_csv}")
    print(f"[AGREEMENT] Wrote {output_aggregate_csv}")
    print(f"[AGREEMENT] Wrote {output_pairwise_csv}")
    print(f"[AGREEMENT] Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
