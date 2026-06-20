#!/usr/bin/env python3
"""Compute BLEU and ROUGE metrics for generated PR descriptions.

This script uses original PR descriptions as references and computes:
- corpus BLEU (NLTK, smoothing method1)
- ROUGE-1 F1
- ROUGE-2 F1
- ROUGE-L F1

Metrics are reported per dataset and generation mode.
"""

from __future__ import annotations

import csv
import json
import math
import re
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
OUT_PREFIX = "lexical_metrics_summary"
DELTA_OUT_PREFIX = "lexical_metrics_delta_from_raw"
JUDGE_SUMMARY_CSV = TABLES_DIR / "descriptions_summary.csv"

REQUIRED_MODES = {"raw", "cmg_only", "file_summaries_only", "full"}
MODE_ORDER = {"raw": 0, "cmg_only": 1, "file_summaries_only": 2, "full": 3}
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


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: List[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n_f1(reference_tokens: List[str], candidate_tokens: List[str], n: int) -> float:
    ref = ngrams(reference_tokens, n)
    cand = ngrams(candidate_tokens, n)
    if not ref or not cand:
        return 0.0
    overlap = 0
    for gram, count in cand.items():
        overlap += min(count, ref.get(gram, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / sum(cand.values())
    recall = overlap / sum(ref.values())
    if precision + recall == 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0]
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(reference_tokens: List[str], candidate_tokens: List[str]) -> float:
    if not reference_tokens or not candidate_tokens:
        return 0.0
    lcs = lcs_len(reference_tokens, candidate_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def corpus_bleu_4(
    references: List[List[str]],
    hypotheses: List[List[str]],
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> float:
    if not references or not hypotheses or len(references) != len(hypotheses):
        return 0.0

    matches_by_order = [0, 0, 0, 0]
    possible_by_order = [0, 0, 0, 0]
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)
        for n in range(1, 5):
            ref_ngrams = ngrams(ref, n)
            hyp_ngrams = ngrams(hyp, n)
            overlap = 0
            for gram, count in hyp_ngrams.items():
                overlap += min(count, ref_ngrams.get(gram, 0))
            matches_by_order[n - 1] += overlap
            possible_by_order[n - 1] += max(len(hyp) - n + 1, 0)

    precisions: List[float] = []
    for i in range(4):
        # Add-one smoothing to avoid zeroing the full geometric mean on sparse text.
        precisions.append((matches_by_order[i] + 1.0) / (possible_by_order[i] + 1.0))

    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len))
    s = 0.0
    for w, p in zip(weights, precisions):
        s += w * math.log(max(p, 1e-12))
    return bp * math.exp(s)


def safe_mean(values: List[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def safe_stdev(values: List[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def collect_metrics(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Tuple[List[str], List[str], float, float, float]]] = defaultdict(list)
    bleu_refs: Dict[str, List[List[str]]] = defaultdict(list)
    bleu_hyps: Dict[str, List[List[str]]] = defaultdict(list)

    for rec in records:
        mode = str(rec.get("generation_mode") or "")
        if mode not in REQUIRED_MODES:
            continue
        reference_text = str(rec.get("original_description") or "")
        candidate_text = str(rec.get("generated_description") or "")
        ref_toks = tokenize(reference_text)
        cand_toks = tokenize(candidate_text)
        if not ref_toks or not cand_toks:
            continue

        r1 = rouge_n_f1(ref_toks, cand_toks, 1)
        r2 = rouge_n_f1(ref_toks, cand_toks, 2)
        rl = rouge_l_f1(ref_toks, cand_toks)
        grouped[mode].append((ref_toks, cand_toks, r1, r2, rl))
        bleu_refs[mode].append(ref_toks)
        bleu_hyps[mode].append(cand_toks)

    out: Dict[str, Dict[str, Any]] = {}
    for mode in sorted(grouped.keys(), key=lambda m: MODE_ORDER.get(m, 99)):
        rows = grouped[mode]
        r1_vals = [x[2] for x in rows]
        r2_vals = [x[3] for x in rows]
        rl_vals = [x[4] for x in rows]
        bleu = corpus_bleu_4(bleu_refs[mode], bleu_hyps[mode])
        out[mode] = {
            "n": len(rows),
            "bleu": bleu,
            "rouge_1_f1": safe_mean(r1_vals),
            "rouge_2_f1": safe_mean(r2_vals),
            "rouge_l_f1": safe_mean(rl_vals),
            "rouge_l_f1_stdev": safe_stdev(rl_vals),
        }
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "mode",
        "n",
        "bleu",
        "rouge_1_f1",
        "rouge_2_f1",
        "rouge_l_f1",
        "rouge_l_f1_stdev",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_delta_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "mode",
        "delta_bleu_vs_raw",
        "delta_rouge_l_f1_vs_raw",
        "delta_judge_mean_vs_raw",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_md(path: Path, dataset_rows: Dict[str, List[Dict[str, Any]]]) -> None:
    lines: List[str] = ["# Lexical Metrics Summary", ""]
    lines.append(
        "Reference text is the original PR description; candidate text is the generated description for each mode."
    )
    lines.append("")

    for dataset in sorted(dataset_rows.keys(), key=lambda d: DATASET_ORDER.get(d, 99)):
        lines.append(f"## {DATASET_LABELS.get(dataset, dataset.upper())}")
        lines.append("")
        lines.append("| mode | n | BLEU | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | ROUGE-L SD |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in dataset_rows[dataset]:
            lines.append(
                "| {mode} | {n} | {bleu} | {rouge_1_f1} | {rouge_2_f1} | {rouge_l_f1} | {rouge_l_f1_stdev} |".format(
                    **{k: fmt(v) for k, v in r.items()}
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_delta_md(path: Path, dataset_rows: Dict[str, List[Dict[str, Any]]]) -> None:
    lines: List[str] = ["# Lexical/Judge Delta from Raw", ""]
    lines.append(
        "Deltas are computed relative to `raw` within each dataset. Positive `delta_judge_mean_vs_raw` indicates improved judge score over raw."
    )
    lines.append("")
    for dataset in sorted(dataset_rows.keys(), key=lambda d: DATASET_ORDER.get(d, 99)):
        lines.append(f"## {DATASET_LABELS.get(dataset, dataset.upper())}")
        lines.append("")
        lines.append("| mode | delta BLEU vs raw | delta ROUGE-L F1 vs raw | delta judge mean vs raw |")
        lines.append("| --- | --- | --- | --- |")
        for r in dataset_rows[dataset]:
            lines.append(
                "| {mode} | {delta_bleu_vs_raw} | {delta_rouge_l_f1_vs_raw} | {delta_judge_mean_vs_raw} |".format(
                    **{k: fmt(v) for k, v in r.items()}
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_judge_means() -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    if not JUDGE_SUMMARY_CSV.exists():
        return out
    with JUDGE_SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dataset = (row.get("dataset") or "").strip().lower()
            mode = (row.get("mode") or "").strip()
            if not dataset or mode not in REQUIRED_MODES:
                continue
            try:
                out[(dataset, mode)] = float(row.get("avg_generated_score") or "")
            except Exception:
                continue
    return out


def build_delta_from_raw(
    dataset_rows: Dict[str, List[Dict[str, Any]]],
    judge_means: Dict[Tuple[str, str], float],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    all_rows: List[Dict[str, Any]] = []
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for dataset, rows in dataset_rows.items():
        by_mode = {str(r.get("mode")): r for r in rows}
        raw = by_mode.get("raw")
        if not raw:
            continue
        raw_bleu = float(raw.get("bleu") or 0.0)
        raw_rl = float(raw.get("rouge_l_f1") or 0.0)
        raw_judge = judge_means.get((dataset, "raw"), 0.0)

        ds_rows: List[Dict[str, Any]] = []
        for mode in ("raw", "cmg_only", "file_summaries_only", "full"):
            curr = by_mode.get(mode)
            if not curr:
                continue
            bleu = float(curr.get("bleu") or 0.0)
            rl = float(curr.get("rouge_l_f1") or 0.0)
            judge = judge_means.get((dataset, mode), 0.0)
            rec = {
                "dataset": dataset,
                "mode": mode,
                "delta_bleu_vs_raw": bleu - raw_bleu,
                "delta_rouge_l_f1_vs_raw": rl - raw_rl,
                "delta_judge_mean_vs_raw": judge - raw_judge,
            }
            ds_rows.append(rec)
            all_rows.append(rec)
        by_dataset[dataset] = ds_rows
    return all_rows, by_dataset


def sync_results_to_research_paper() -> Path | None:
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

    summary_json: Dict[str, Any] = {"datasets": {}, "inputs": [p.name for p in files]}
    csv_rows: List[Dict[str, Any]] = []
    dataset_rows: Dict[str, List[Dict[str, Any]]] = {}

    for dataset, records in by_dataset_raw.items():
        filtered = filter_complete_prs(records)
        deduped = dedupe_by_pr_mode(filtered)
        metrics = collect_metrics(deduped)

        table_rows: List[Dict[str, Any]] = []
        for mode, vals in sorted(metrics.items(), key=lambda x: MODE_ORDER.get(x[0], 99)):
            row = {"dataset": dataset, "mode": mode, **vals}
            table_rows.append(row)
            csv_rows.append(row)

        dataset_rows[dataset] = table_rows
        summary_json["datasets"][dataset] = {
            "mode_metrics": metrics,
            "records_after_filter": len(deduped),
        }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_json = TABLES_DIR / f"{OUT_PREFIX}.json"
    out_csv = TABLES_DIR / f"{OUT_PREFIX}.csv"
    out_md = TABLES_DIR / f"{OUT_PREFIX}.md"
    out_delta_json = TABLES_DIR / f"{DELTA_OUT_PREFIX}.json"
    out_delta_csv = TABLES_DIR / f"{DELTA_OUT_PREFIX}.csv"
    out_delta_md = TABLES_DIR / f"{DELTA_OUT_PREFIX}.md"

    out_json.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    write_csv(out_csv, csv_rows)
    write_md(out_md, dataset_rows)

    judge_means = load_judge_means()
    delta_rows, delta_by_dataset = build_delta_from_raw(dataset_rows, judge_means)
    delta_json_obj = {
        "datasets": {
            ds: {"rows": rows}
            for ds, rows in delta_by_dataset.items()
        },
        "judge_summary_source": str(JUDGE_SUMMARY_CSV) if JUDGE_SUMMARY_CSV.exists() else None,
    }
    out_delta_json.write_text(json.dumps(delta_json_obj, indent=2), encoding="utf-8")
    write_delta_csv(out_delta_csv, delta_rows)
    write_delta_md(out_delta_md, delta_by_dataset)

    synced_dir = sync_results_to_research_paper()

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_delta_json}")
    print(f"Wrote {out_delta_csv}")
    print(f"Wrote {out_delta_md}")
    if synced_dir is not None:
        print(f"Synced final-results tables to {synced_dir}")
    else:
        print("Skipped sync: research-paper directory not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
