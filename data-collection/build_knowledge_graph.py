#!/usr/bin/env python3
"""
Batch collector that builds a pull-request knowledge graph from entries in the configured dataset CSV.
"""

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_COLLECTION_DIR = Path(__file__).resolve().parent
PR_STAGE_DIR = ROOT_DIR / "description-generation"
for path in (ROOT_DIR, DATA_COLLECTION_DIR, PR_STAGE_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from dotenv import load_dotenv

from config.loader import load_pipeline_config
from knowledge_graph import KnowledgeGraphBuilder
from orchestrator.pr_orchestrator import PRDescriptionOrchestrator


def read_pr_targets(csv_path: Path, limit: int | None = None) -> Iterator[Tuple[str, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        count = 0
        for row in reader:
            repo_name = (row.get("repo_name") or row.get("repo") or "").strip()
            pr_str = (row.get("pr_number") or row.get("pr") or "").strip()
            if not repo_name or not pr_str:
                continue

            try:
                pr_number = int(pr_str)
            except ValueError:
                continue

            yield repo_name, pr_number
            count += 1
            if limit is not None and count >= limit:
                return


def parse_args(default_input: Path, default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct a knowledge graph from pull requests listed in the configured dataset CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"CSV file containing repo_name and pr_number columns (default: {default_input}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to write the knowledge graph in node-link JSON format.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of PRs to process.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize PR order before processing (applies limit after shuffle).",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    pipeline_config = load_pipeline_config()
    dataset_config = pipeline_config.get("dataset", {}) or {}
    dataset_csv = (dataset_config.get("csv_path") or "data/parsed.csv").strip()
    dataset_name = (dataset_config.get("name") or Path(dataset_csv).stem).strip()
    llm_config = pipeline_config.get("llm", {})
    cmg_config = pipeline_config.get("cmg", {})
    ranking_config = pipeline_config.get("ranking", {}) or {}
    env_provider = os.getenv("LLM_PROVIDER")
    llm_provider = (env_provider or llm_config.get("provider") or "openai").lower()

    def log_error(message: str) -> None:
        log_path = ROOT_DIR / "logs" / "knowledge-graph" / "errors.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{message}\n")

    github_token = os.getenv("GITHUB_CLASSIC_TOKEN") or os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise RuntimeError("Missing GITHUB_TOKEN or GITHUB_CLASSIC_TOKEN in environment.")

    default_input = ROOT_DIR / dataset_csv
    default_output = ROOT_DIR / "results" / "knowledge_graph" / f"graph-{dataset_name}.json"
    args = parse_args(default_input, default_output)
    if args.randomize:
        pr_targets = list(read_pr_targets(args.input, None))
        random.shuffle(pr_targets)
        if args.limit is not None:
            pr_targets = pr_targets[: args.limit]
    else:
        pr_targets = list(read_pr_targets(args.input, args.limit))

    if not pr_targets:
        print("[GRAPH] No pull requests found in the provided CSV.")
        return 0

    print(f"[GRAPH] Building knowledge graph for {len(pr_targets)} pull requests...\n")

    orchestrator = PRDescriptionOrchestrator(
        github_token=github_token,
        llm_api_key=None,
        enable_llm_components=False,
        llm_provider=llm_provider,
        llm_settings=llm_config,
        cmg_config=cmg_config,
        ranking_config=ranking_config,
    )
    graph_builder = KnowledgeGraphBuilder()

    successes = 0
    failures = 0

    start_time = time.time()

    def _retry_collect(repo_name: str, pr_number: int, max_attempts: int = 5) -> dict:
        """Retry collection with exponential backoff on rate-limit-like errors."""
        delay = 2.0
        for attempt in range(1, max_attempts + 1):
            try:
                return orchestrator.collect_pr_context(repo_name, pr_number)
            except Exception as exc:
                msg = str(exc).lower()
                rate_limited = any(keyword in msg for keyword in ("rate", "429", "abuse", "secondary"))
                if attempt == max_attempts or not rate_limited:
                    raise
                print(f"[GRAPH][RETRY] Attempt {attempt} hit rate limit for {repo_name}#{pr_number}. Sleeping {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60.0)

    for idx, (repo_name, pr_number) in enumerate(pr_targets, start=1):
        print(f"[GRAPH] ({idx}/{len(pr_targets)}) Collecting {repo_name}#PR{pr_number}...")
        try:
            pr_context = _retry_collect(repo_name, pr_number)
            graph_builder.ingest_pr_data(pr_context)
            graph_builder.save_json(args.output)  # persist after each successful PR
            successes += 1
        except Exception as exc:  # pragma: no cover - diagnostic output
            failures += 1
            msg = f"[GRAPH][ERROR] Failed to process {repo_name} PR #{pr_number}: {exc}"
            print(msg)
            log_error(msg)
            continue

    elapsed = time.time() - start_time
    output_path = graph_builder.save_json(args.output)

    print(
        "\n[GRAPH] Completed knowledge graph build.\n"
        f"         Successes: {successes}\n"
        f"         Failures : {failures}\n"
        f"         Nodes    : {graph_builder.graph.number_of_nodes()}\n"
        f"         Edges    : {graph_builder.graph.number_of_edges()}\n"
        f"         Output   : {output_path}\n"
        f"         Duration : {elapsed:.2f}s\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
