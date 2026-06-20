#!/usr/bin/env python3
"""Analyze LLM-judge penalty justifications for PR description generation.

This script mirrors the data loading/filtering logic in analyze_descriptions.py and
produces reusable failure-analysis artifacts over the full evaluated PR set.

Outputs (in final-results/eval/tables):
- failure_analysis_summary.json
- failure_analysis_summary.csv
- failure_analysis_summary.md
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent
RESEARCH_PAPER_DIR = REPO_ROOT / "research-paper"
PAPER_FINAL_RESULTS_DIR = RESEARCH_PAPER_DIR / "final-results"
DATA_DIR = ROOT / "data" / "description-data"
TABLES_DIR = ROOT / "eval" / "tables"
OUT_PREFIX = "failure_analysis_summary"

REQUIRED_MODES = {"raw", "cmg_only", "file_summaries_only", "full"}
MODE_ORDER = {"raw": 0, "cmg_only": 1, "file_summaries_only": 2, "full": 3}


def infer_dataset(name: str) -> str:
    lowered = name.lower()
    if "aidev" in lowered:
        return "aidev"
    if "parsed" in lowered or "trudeau" in lowered:
        return "prsummarizer"
    return "unknown"


def extract_records(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def list_input_files() -> List[Path]:
    candidates: List[Path] = []
    for p in DATA_DIR.glob("*.json"):
        n = p.name
        if n.startswith("results_") and n.endswith(".json"):
            candidates.append(p)
        elif n.startswith("descriptions-") and "judge" in n and n.endswith(".json"):
            candidates.append(p)
    return sorted(candidates)


def filter_complete_prs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_pr: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        repo = rec.get("repo_name")
        pr = rec.get("pr_number")
        mode = rec.get("generation_mode")
        if not repo or pr is None or not mode:
            continue
        try:
            key = (str(repo), int(pr))
        except Exception:
            continue
        by_pr[key].append(rec)

    kept: List[Dict[str, Any]] = []
    for group in by_pr.values():
        modes = {str(r.get("generation_mode")) for r in group}
        if REQUIRED_MODES.issubset(modes):
            kept.extend(group)
    return kept


def dedupe_by_pr_mode(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, int, str]] = set()
    out: List[Dict[str, Any]] = []
    for rec in records:
        repo = rec.get("repo_name")
        pr = rec.get("pr_number")
        mode = rec.get("generation_mode")
        if not repo or pr is None or not mode:
            continue
        try:
            key = (str(repo), int(pr), str(mode))
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def classify_penalty(reason: str) -> str:
    text = (reason or "").lower()

    if "linked issues" in text or "linked issue" in text:
        return "linked_issue_mismatch"

    if "tests section" in text or (
        "tests" in text and any(k in text for k in ["does not include", "omits", "missing", "not include"])
    ):
        return "missing_test_coverage"

    if any(
        k in text
        for k in [
            "unsupported claim",
            "incorrect",
            "inconsistent",
            "mismatched",
            "appears to preexist",
            "not supported",
        ]
    ):
        return "unsupported_specifics"

    if any(
        k in text
        for k in [
            "missing mention",
            "does not mention",
            "omits",
            "missing coverage",
            "not mentioned",
            "top-changed file not mentioned",
        ]
    ):
        return "missing_key_changes"

    if any(k in text for k in ["format", "structure", "json", "section header"]):
        return "format_structure"

    return "other"


def safe_pct(n: int, d: int) -> float:
    return (n / d) if d else 0.0


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "mode",
        "n",
        "penalized_pr_count",
        "penalized_pr_rate",
        "total_penalties",
        "missing_key_changes_count",
        "missing_key_changes_rate",
        "unsupported_specifics_count",
        "unsupported_specifics_rate",
        "missing_test_coverage_count",
        "missing_test_coverage_rate",
        "linked_issue_mismatch_count",
        "linked_issue_mismatch_rate",
        "format_structure_count",
        "format_structure_rate",
        "other_count",
        "other_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt_num(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def fmt_pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except Exception:
        return "n/a"


def write_md(path: Path, rows: List[Dict[str, Any]], combined: Dict[str, Any], raw_to_full: Dict[str, Dict[str, int]]) -> None:
    lines: List[str] = ["# Failure Analysis Summary", ""]
    lines.append("## Key Findings")
    lines.append(
        f"- Full evaluated set size per mode: `{rows[0]['n']}` PRs (across AIDev + PRSummarizer-derived after complete-PR filtering and dedupe)."
    )
    lines.append(
        f"- Penalized PR rate drops from `{fmt_pct(rows[0]['penalized_pr_rate'])}` in `raw` to `{fmt_pct(rows[-1]['penalized_pr_rate'])}` in `full`."
    )
    lines.append(
        f"- Dominant failure type overall: `missing_key_changes` at `{fmt_pct(combined['missing_key_changes_rate'])}` of all penalties."
    )
    lines.append("")

    lines.append("## By Mode")
    lines.append("")
    lines.append("| mode | n | penalized PRs | penalized rate | penalties | missing key changes | unsupported specifics | missing test coverage | linked issue mismatch | format/structure | other |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            "| {mode} | {n} | {penalized_pr_count} | {penalized_pr_rate} | {total_penalties} | {mk} ({mkp}) | {us} ({usp}) | {mt} ({mtp}) | {li} ({lip}) | {fs} ({fsp}) | {ot} ({otp}) |".format(
                mode=r["mode"],
                n=r["n"],
                penalized_pr_count=r["penalized_pr_count"],
                penalized_pr_rate=fmt_pct(r["penalized_pr_rate"]),
                total_penalties=r["total_penalties"],
                mk=r["missing_key_changes_count"],
                mkp=fmt_pct(r["missing_key_changes_rate"]),
                us=r["unsupported_specifics_count"],
                usp=fmt_pct(r["unsupported_specifics_rate"]),
                mt=r["missing_test_coverage_count"],
                mtp=fmt_pct(r["missing_test_coverage_rate"]),
                li=r["linked_issue_mismatch_count"],
                lip=fmt_pct(r["linked_issue_mismatch_rate"]),
                fs=r["format_structure_count"],
                fsp=fmt_pct(r["format_structure_rate"]),
                ot=r["other_count"],
                otp=fmt_pct(r["other_rate"]),
            )
        )
    lines.append("")

    lines.append("## Raw to Full Category Deltas")
    lines.append("")
    lines.append("| category | raw | full | delta (full - raw) |")
    lines.append("| --- | --- | --- | --- |")
    for cat, vals in raw_to_full.items():
        lines.append(f"| {cat} | {vals['raw']} | {vals['full']} | {vals['delta']} |")

    path.write_text("\n".join(lines), encoding="utf-8")


def sync_tables_to_research_paper() -> Path | None:
    if not RESEARCH_PAPER_DIR.exists():
        return None

    PAPER_FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tables_src = ROOT / "eval" / "tables"
    tables_dst = PAPER_FINAL_RESULTS_DIR / "tables"
    if tables_src.exists():
        shutil.copytree(tables_src, tables_dst, dirs_exist_ok=True)
    return PAPER_FINAL_RESULTS_DIR


def main() -> int:
    files = list_input_files()
    if not files:
        raise SystemExit("No description JSON files found.")

    by_dataset_raw: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in files:
        dataset = infer_dataset(path.name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        by_dataset_raw[dataset].extend(list(extract_records(payload)))

    merged: List[Dict[str, Any]] = []
    dataset_counts: Dict[str, int] = {}

    for dataset, records in by_dataset_raw.items():
        filtered = filter_complete_prs(records)
        deduped = dedupe_by_pr_mode(filtered)
        dataset_counts[dataset] = len(deduped)
        merged.extend(deduped)

    mode_penalty_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    mode_penalty_totals: Counter[str] = Counter()
    mode_pr_totals: Counter[str] = Counter()
    mode_pr_penalized: Counter[str] = Counter()

    for rec in merged:
        mode = str(rec.get("generation_mode") or "")
        if mode not in REQUIRED_MODES:
            continue

        mode_pr_totals[mode] += 1

        judgment = rec.get("judgment") or {}
        gb = judgment.get("generated_breakdown") or {}
        penalties = gb.get("penalties") or []
        if not isinstance(penalties, list):
            penalties = []

        if penalties:
            mode_pr_penalized[mode] += 1

        for p in penalties:
            if not isinstance(p, str):
                continue
            cat = classify_penalty(p)
            mode_penalty_counts[mode][cat] += 1
            mode_penalty_totals[mode] += 1

    csv_rows: List[Dict[str, Any]] = []
    combined = Counter()
    for mode in sorted(REQUIRED_MODES, key=lambda m: MODE_ORDER[m]):
        total = int(mode_penalty_totals[mode])
        n = int(mode_pr_totals[mode])
        row = {
            "mode": mode,
            "n": n,
            "penalized_pr_count": int(mode_pr_penalized[mode]),
            "penalized_pr_rate": safe_pct(int(mode_pr_penalized[mode]), n),
            "total_penalties": total,
            "missing_key_changes_count": int(mode_penalty_counts[mode]["missing_key_changes"]),
            "missing_key_changes_rate": safe_pct(int(mode_penalty_counts[mode]["missing_key_changes"]), total),
            "unsupported_specifics_count": int(mode_penalty_counts[mode]["unsupported_specifics"]),
            "unsupported_specifics_rate": safe_pct(int(mode_penalty_counts[mode]["unsupported_specifics"]), total),
            "missing_test_coverage_count": int(mode_penalty_counts[mode]["missing_test_coverage"]),
            "missing_test_coverage_rate": safe_pct(int(mode_penalty_counts[mode]["missing_test_coverage"]), total),
            "linked_issue_mismatch_count": int(mode_penalty_counts[mode]["linked_issue_mismatch"]),
            "linked_issue_mismatch_rate": safe_pct(int(mode_penalty_counts[mode]["linked_issue_mismatch"]), total),
            "format_structure_count": int(mode_penalty_counts[mode]["format_structure"]),
            "format_structure_rate": safe_pct(int(mode_penalty_counts[mode]["format_structure"]), total),
            "other_count": int(mode_penalty_counts[mode]["other"]),
            "other_rate": safe_pct(int(mode_penalty_counts[mode]["other"]), total),
        }
        csv_rows.append(row)
        combined += mode_penalty_counts[mode]

    combined_total = int(sum(combined.values()))
    combined_stats = {
        f"{cat}_count": int(combined[cat])
        for cat in [
            "missing_key_changes",
            "unsupported_specifics",
            "missing_test_coverage",
            "linked_issue_mismatch",
            "format_structure",
            "other",
        ]
    }
    combined_stats.update(
        {
            f"{cat}_rate": safe_pct(int(combined[cat]), combined_total)
            for cat in [
                "missing_key_changes",
                "unsupported_specifics",
                "missing_test_coverage",
                "linked_issue_mismatch",
                "format_structure",
                "other",
            ]
        }
    )
    combined_stats["total_penalties"] = combined_total

    raw_row = next((r for r in csv_rows if r["mode"] == "raw"), None)
    full_row = next((r for r in csv_rows if r["mode"] == "full"), None)
    categories = [
        "missing_key_changes",
        "unsupported_specifics",
        "missing_test_coverage",
        "linked_issue_mismatch",
        "format_structure",
        "other",
    ]
    raw_to_full: Dict[str, Dict[str, int]] = {}
    for cat in categories:
        raw_v = int(raw_row[f"{cat}_count"]) if raw_row else 0
        full_v = int(full_row[f"{cat}_count"]) if full_row else 0
        raw_to_full[cat] = {"raw": raw_v, "full": full_v, "delta": full_v - raw_v}

    summary = {
        "inputs": [p.name for p in files],
        "dataset_records_after_filter": dataset_counts,
        "per_mode": csv_rows,
        "combined": combined_stats,
        "raw_to_full": raw_to_full,
    }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_json = TABLES_DIR / f"{OUT_PREFIX}.json"
    out_csv = TABLES_DIR / f"{OUT_PREFIX}.csv"
    out_md = TABLES_DIR / f"{OUT_PREFIX}.md"

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_csv, csv_rows)
    write_md(out_md, csv_rows, combined_stats, raw_to_full)
    synced_dir = sync_tables_to_research_paper()

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    if synced_dir is not None:
        print(f"Synced tables to {synced_dir}")
    else:
        print("Skipped sync: research-paper directory not found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
