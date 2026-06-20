#!/usr/bin/env python3
"""
Create a survey-ready export from a judge results file.
Outputs rows with original and generated descriptions per mode.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime
import re


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list in judge results file.")
    return data


def _group_by_pr(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        repo = row.get("repo_name")
        pr_number = row.get("pr_number")
        if not repo or pr_number is None:
            continue
        key = (repo, int(pr_number))
        grouped.setdefault(key, []).append(row)
    return grouped


def _mode_key(mode: str) -> str:
    return (mode or "").strip().lower()

def _dataset_from_filename(path: Path) -> str:
    match = re.search(r"descriptions-([a-z0-9_-]+)-", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return "dataset"


def _extract_original_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    judgment = row.get("judgment") or {}
    original_breakdown = judgment.get("original_breakdown") or {}
    analysis = row.get("analysis") or {}
    return {
        "original_description": row.get("original_description"),
        "original_score": judgment.get("original_score"),
        "original_correctness_penalty": original_breakdown.get("correctness_penalty"),
        "original_coverage_penalty": original_breakdown.get("coverage_penalty"),
        "original_clarity_penalty": original_breakdown.get("clarity_penalty"),
        "original_primary_reason": analysis.get("original_primary_reason"),
        "original_penalties": json.dumps(original_breakdown.get("penalties") or [], ensure_ascii=True),
    }


def _extract_generated_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    judgment = row.get("judgment") or {}
    generated_breakdown = judgment.get("generated_breakdown") or {}
    analysis = row.get("analysis") or {}
    return {
        "generated_description": row.get("generated_description"),
        "generated_score": judgment.get("generated_score"),
        "generated_correctness_penalty": generated_breakdown.get("correctness_penalty"),
        "generated_coverage_penalty": generated_breakdown.get("coverage_penalty"),
        "generated_clarity_penalty": generated_breakdown.get("clarity_penalty"),
        "generated_primary_reason": analysis.get("generated_primary_reason"),
        "generated_penalties": json.dumps(generated_breakdown.get("penalties") or [], ensure_ascii=True),
    }


def _write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = _group_by_pr(rows)
    mode_order = ["raw", "cmg_only", "file_summaries_only", "full"]
    fieldnames = [
        "repo_name",
        "pr_number",
        "original_description",
        "original_score",
        "original_correctness_penalty",
        "original_coverage_penalty",
        "original_clarity_penalty",
        "original_primary_reason",
        "original_penalties",
    ]
    for mode in mode_order:
        fieldnames.extend(
            [
                f"{mode}_description",
                f"{mode}_score",
                f"{mode}_correctness_penalty",
                f"{mode}_coverage_penalty",
                f"{mode}_clarity_penalty",
                f"{mode}_primary_reason",
                f"{mode}_penalties",
            ]
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (repo, pr_number), group_rows in grouped.items():
            base_row: Dict[str, Any] = {
                "repo_name": repo,
                "pr_number": pr_number,
            }
            original_row = group_rows[0]
            base_row.update(_extract_original_fields(original_row))
            for mode in mode_order:
                for row in group_rows:
                    if _mode_key(row.get("generation_mode")) == mode:
                        gen = _extract_generated_fields(row)
                        base_row[f"{mode}_description"] = gen["generated_description"]
                        base_row[f"{mode}_score"] = gen["generated_score"]
                        base_row[f"{mode}_correctness_penalty"] = gen["generated_correctness_penalty"]
                        base_row[f"{mode}_coverage_penalty"] = gen["generated_coverage_penalty"]
                        base_row[f"{mode}_clarity_penalty"] = gen["generated_clarity_penalty"]
                        base_row[f"{mode}_primary_reason"] = gen["generated_primary_reason"]
                        base_row[f"{mode}_penalties"] = gen["generated_penalties"]
                        break
            writer.writerow(
                base_row
            )


def _write_json(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    minimal = []
    grouped = _group_by_pr(rows)
    mode_order = ["raw", "cmg_only", "file_summaries_only", "full"]
    for (repo, pr_number), group_rows in grouped.items():
        original_row = group_rows[0]
        original_fields = _extract_original_fields(original_row)
        base = {
            "repo_name": repo,
            "pr_number": pr_number,
            "original": {
                "mode": "original",
                "description": original_fields.get("original_description"),
                "score": original_fields.get("original_score"),
                "correctness_penalty": original_fields.get("original_correctness_penalty"),
                "coverage_penalty": original_fields.get("original_coverage_penalty"),
                "clarity_penalty": original_fields.get("original_clarity_penalty"),
                "primary_reason": original_fields.get("original_primary_reason"),
                "penalties": original_fields.get("original_penalties"),
            },
        }
        for mode in mode_order:
            for row in group_rows:
                if _mode_key(row.get("generation_mode")) == mode:
                    gen = _extract_generated_fields(row)
                    base[f"generated_{mode}"] = {
                        "mode": mode,
                        "description": gen.get("generated_description"),
                        "score": gen.get("generated_score"),
                        "correctness_penalty": gen.get("generated_correctness_penalty"),
                        "coverage_penalty": gen.get("generated_coverage_penalty"),
                        "clarity_penalty": gen.get("generated_clarity_penalty"),
                        "primary_reason": gen.get("generated_primary_reason"),
                        "penalties": gen.get("generated_penalties"),
                    }
                    break
        minimal.append(base)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(minimal, handle, indent=2)


def main() -> int:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create a survey export from a judge results JSON file."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Judge results filename or path (e.g., descriptions-...-judge-....json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: results/survey/survey-<dataset>-<timestamp>.json).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute() and not input_path.exists():
        input_path = root_dir / "results" / "judge" / "openai" / input_path.name
    rows = _load_json(input_path)

    output_path = args.output
    if output_path is None:
        dataset = _dataset_from_filename(input_path)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        output_path = root_dir / "results" / "survey" / f"survey-{dataset}-{timestamp}.json"

    _write_json(rows, output_path)

    print(f"[SURVEY] Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
