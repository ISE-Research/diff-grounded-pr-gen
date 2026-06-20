#!/usr/bin/env python3
"""Analyze generated-description judge JSON artifacts.

This script summarizes the six description result JSON files and outputs
per-dataset mode metrics (including a synthetic original baseline row).

Outputs:
- descriptions_summary.json
- descriptions_summary.csv
- descriptions_summary.md
"""

from __future__ import annotations

import csv
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent
RESEARCH_PAPER_DIR = REPO_ROOT / "research-paper"
PAPER_FINAL_RESULTS_DIR = RESEARCH_PAPER_DIR / "final-results"
DATA_DIR = ROOT / "data" / "description-data"
TABLES_DIR = ROOT / "eval" / "tables"
PLOTS_DIR = ROOT / "eval" / "plots"
OUT_PREFIX = "descriptions_summary"
REQUIRED_MODES = {"raw", "cmg_only", "file_summaries_only", "full"}
DATASET_ORDER = {"aidev": 0, "prsummarizer": 1}
DATASET_LABELS = {"aidev": "AIDev", "prsummarizer": "PRSummarizer"}


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
    # Keep first occurrence by (repo, pr, mode) after file-order concatenation.
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


def build_original_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()
    for rec in records:
        repo = rec.get("repo_name")
        pr = rec.get("pr_number")
        judgment = rec.get("judgment") or {}
        if not repo or pr is None or not isinstance(judgment, dict):
            continue
        try:
            key = (str(repo), int(pr))
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "repo_name": key[0],
                "pr_number": key[1],
                "generation_mode": "original",
                "judgment": {
                    "generated_score": judgment.get("original_score"),
                    "original_score": judgment.get("original_score"),
                    "prefers": "original",
                    "generated_breakdown": judgment.get("original_breakdown") or {},
                },
            }
        )
    return out


def safe_mean(values: List[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def safe_median(values: List[float]) -> float | None:
    return statistics.median(values) if values else None


def safe_stdev(values: List[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def to_float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def collect_mode_stats(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    pref_generated_counts: Dict[str, int] = defaultdict(int)
    total_counts: Dict[str, int] = defaultdict(int)

    for rec in records:
        mode = str(rec.get("generation_mode") or "unknown")
        judgment = rec.get("judgment") or {}
        if not isinstance(judgment, dict):
            continue
        try:
            g = float(judgment.get("generated_score"))
            o = float(judgment.get("original_score"))
        except Exception:
            continue

        gb = judgment.get("generated_breakdown") or {}
        corr = to_float_or_default(gb.get("correctness_penalty"), 0.0)
        cov = to_float_or_default(gb.get("coverage_penalty"), 0.0)
        clar = to_float_or_default(gb.get("clarity_penalty"), 0.0)

        buckets[mode]["generated"].append(g)
        buckets[mode]["original"].append(o)
        buckets[mode]["delta"].append(g - o)
        buckets[mode]["correctness"].append(corr)
        buckets[mode]["coverage"].append(cov)
        buckets[mode]["clarity"].append(clar)

        total_counts[mode] += 1
        if judgment.get("prefers") == "generated":
            pref_generated_counts[mode] += 1

    stats: Dict[str, Dict[str, Any]] = {}
    for mode, vals in buckets.items():
        n = total_counts[mode]
        stats[mode] = {
            "count": n,
            "avg_generated_score": safe_mean(vals["generated"]),
            "avg_original_score": safe_mean(vals["original"]),
            "avg_delta": safe_mean(vals["delta"]),
            "median_generated_score": safe_median(vals["generated"]),
            "stdev_generated_score": safe_stdev(vals["generated"]),
            "pref_generated_rate": (pref_generated_counts[mode] / n) if n else None,
            "avg_correctness_penalty": safe_mean(vals["correctness"]),
            "avg_coverage_penalty": safe_mean(vals["coverage"]),
            "avg_clarity_penalty": safe_mean(vals["clarity"]),
        }
    return stats


def winner_by_avg(stats: Dict[str, Dict[str, Any]]) -> str | None:
    best_mode = None
    best_val = None
    for mode, s in stats.items():
        v = s.get("avg_generated_score")
        if v is None:
            continue
        if best_val is None or v > best_val:
            best_val = v
            best_mode = mode
    return best_mode


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "mode",
        "n",
        "avg_generated_score",
        "avg_original_score",
        "avg_delta",
        "median_generated_score",
        "stdev_generated_score",
        "pref_generated_rate",
        "avg_correctness_penalty",
        "avg_coverage_penalty",
        "avg_clarity_penalty",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "n/a"


def find_mode_row(rows: List[Dict[str, Any]], mode: str) -> Dict[str, Any] | None:
    for row in rows:
        if row.get("mode") == mode:
            return row
    return None


def write_md(path: Path, dataset_rows: Dict[str, List[Dict[str, Any]]], winners: Dict[str, str | None]) -> None:
    lines: List[str] = ["# Description Evaluation Summary", ""]
    lines.append("## Key Findings")
    for dataset in sorted(dataset_rows.keys(), key=lambda d: DATASET_ORDER.get(d, 99)):
        rows = dataset_rows[dataset]
        winner = winners.get(dataset)
        winner_row = find_mode_row(rows, str(winner)) if winner else None
        original_row = find_mode_row(rows, "original")
        dataset_label = DATASET_LABELS.get(dataset, dataset.upper())
        if winner_row and original_row:
            lines.append(
                f"- `{dataset_label}` winner: `{winner}` with avg judge score `{fmt(winner_row.get('avg_generated_score'))}` vs original `{fmt(original_row.get('avg_generated_score'))}`."
            )
            lines.append(
                f"- `{dataset_label}` judge prefers generated for `{winner}` at `{pct(winner_row.get('pref_generated_rate'))}`."
            )
    lines.append("")

    for dataset in sorted(dataset_rows.keys(), key=lambda d: DATASET_ORDER.get(d, 99)):
        lines.append(f"## {DATASET_LABELS.get(dataset, dataset.upper())}")
        lines.append("")
        lines.append("| mode | n | avg generated | avg original | avg delta | median generated | stdev generated | pref generated | corr pen | cov pen | clar pen |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in dataset_rows[dataset]:
            lines.append(
                "| {mode} | {n} | {avg_generated_score} | {avg_original_score} | {avg_delta} | {median_generated_score} | {stdev_generated_score} | {pref_generated_rate} | {avg_correctness_penalty} | {avg_coverage_penalty} | {avg_clarity_penalty} |".format(
                    **{k: fmt(v) for k, v in r.items()}
                )
            )
        lines.append("")
        lines.append(f"Winner by avg generated score: `{winners.get(dataset)}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_descriptions_plot(path: Path, dataset_rows: Dict[str, List[Dict[str, Any]]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode_order = ["original", "raw", "cmg_only", "file_summaries_only", "full"]
    labels = {
        "original": "Original",
        "raw": "Raw",
        "cmg_only": "CMG",
        "file_summaries_only": "FileSum",
        "full": "Full",
    }
    colors = {"aidev": "#1f77b4", "prsummarizer": "#ff7f0e"}
    dataset_keys = [d for d in ("aidev", "prsummarizer") if d in dataset_rows]
    metrics = [
        ("avg_generated_score", "Avg Judge Score (higher better)", 0.0, 5.0),
        ("avg_coverage_penalty", "Coverage Penalty (lower better)", 0.0, None),
        ("pref_generated_rate", "Judge Prefers Generated Rate", 0.0, 1.0),
    ]

    index: Dict[Tuple[str, str], Dict[str, Any]] = {
        (dataset, row["mode"]): row
        for dataset, rows in dataset_rows.items()
        for row in rows
    }
    trudeau_original = find_mode_row(dataset_rows.get("prsummarizer", []), "original")
    aidev_original = find_mode_row(dataset_rows.get("aidev", []), "original")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.0), constrained_layout=False)
    fig.suptitle("Description Evaluation by Mode", fontsize=14, y=0.97)

    x = list(range(len(mode_order)))
    n_ds = max(1, len(dataset_keys))
    width = 0.36 if n_ds == 2 else 0.6

    for ax, (metric_key, metric_title, y_min, y_max_fixed) in zip(axes, metrics):
        vals_flat: List[float] = []
        for ds_idx, ds in enumerate(dataset_keys):
            offset = (ds_idx - (n_ds - 1) / 2) * width
            vals = []
            for mode in mode_order:
                row = index.get((ds, mode))
                v = float(row[metric_key]) if row and row.get(metric_key) is not None else 0.0
                vals.append(v)
                vals_flat.append(v)
            ax.bar([i + offset for i in x], vals, width=width, color=colors[ds], label=DATASET_LABELS.get(ds, ds))

        ymax = y_max_fixed if y_max_fixed is not None else (max(vals_flat) * 1.15 if vals_flat else 1.0)
        if ymax <= y_min:
            ymax = y_min + 1.0
        ax.set_ylim(y_min, ymax)
        ax.set_title(metric_title, fontsize=12)
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

    summary_json: Dict[str, Any] = {"datasets": {}, "inputs": [p.name for p in files]}
    csv_rows: List[Dict[str, Any]] = []
    dataset_rows: Dict[str, List[Dict[str, Any]]] = {}
    winners: Dict[str, str | None] = {}

    mode_order = {"original": 0, "raw": 1, "cmg_only": 2, "file_summaries_only": 3, "full": 4}

    for dataset, records in by_dataset_raw.items():
        filtered = filter_complete_prs(records)
        deduped = dedupe_by_pr_mode(filtered)
        with_original = deduped + build_original_rows(deduped)

        stats = collect_mode_stats(with_original)
        winner = winner_by_avg(stats)
        winners[dataset] = winner

        dataset_table: List[Dict[str, Any]] = []
        for mode, s in sorted(stats.items(), key=lambda x: mode_order.get(x[0], 99)):
            row = {"dataset": dataset, "mode": mode, "n": s["count"], **s}
            # normalize key for csv
            row["n"] = row.pop("count") if "count" in row else row["n"]
            dataset_table.append(row)
            csv_rows.append(row)

        dataset_rows[dataset] = dataset_table
        summary_json["datasets"][dataset] = {
            "winner_by_avg_generated_score": winner,
            "mode_stats": stats,
            "records_after_filter": len(deduped),
        }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    out_json = TABLES_DIR / f"{OUT_PREFIX}.json"
    out_csv = TABLES_DIR / f"{OUT_PREFIX}.csv"
    out_md = TABLES_DIR / f"{OUT_PREFIX}.md"
    out_png = PLOTS_DIR / f"{OUT_PREFIX}.png"

    out_json.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    write_csv(out_csv, csv_rows)
    write_md(out_md, dataset_rows, winners)
    written_plot = write_descriptions_plot(out_png, dataset_rows)
    synced_dir = sync_results_to_research_paper()

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(f"Wrote {written_plot}")
    if synced_dir is not None:
        print(f"Synced final-results to {synced_dir}")
    else:
        print("Skipped sync: research-paper directory not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
