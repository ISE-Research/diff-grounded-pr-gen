# main.py
# Entry point to run the PR description generation workflow.

import os
import json
import re
import argparse
import sys
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
DATA_COLLECTION_DIR = ROOT_DIR / "data-collection"
for path in (ROOT_DIR, CURRENT_DIR, DATA_COLLECTION_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from dotenv import load_dotenv

from config.loader import load_pipeline_config

from knowledge_graph import KnowledgeGraphReader
from components.ranking import is_noise_path
from components.file_diff_summarizer import _is_docs_file
from components.commit_message_rewriter import CommitMessageRewriter
from components.ranking import compute_file_scores, rank_commits
from components.file_diff_summarizer import FileDiffSummarizer

load_dotenv()

from orchestrator.pr_orchestrator import PRDescriptionOrchestrator

def _log_error(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{message}\n")

pipeline_config = load_pipeline_config()
dataset_config = pipeline_config.get("dataset", {}) or {}
dataset_csv = (dataset_config.get("csv_path") or "data/parsed.csv").strip()
dataset_name = (dataset_config.get("name") or Path(dataset_csv).stem).strip()
llm_config = pipeline_config.get("llm", {})
cmg_config = pipeline_config.get("cmg", {})
file_summary_config = pipeline_config.get("file_summaries", {}) or {}
ranking_config = pipeline_config.get("ranking", {}) or {}
commit_payload_config = pipeline_config.get("commit_payload", {}) or {}
ranking_config = {
    **ranking_config,
    "commit_payload": commit_payload_config,
}
generation_modes = pipeline_config.get("generation_modes", [])
MAX_PRINT_DIFF_LINES = 5

batch_size_small = file_summary_config.get("batch_size_small")
batch_size_large = file_summary_config.get("batch_size_large")
docs_top_k = file_summary_config.get("docs_top_k")
max_diff_lines_per_prompt = file_summary_config.get("max_diff_lines_per_prompt")
if batch_size_small is not None:
    try:
        FileDiffSummarizer.BATCH_SIZE_SMALL = max(1, int(batch_size_small))
    except (TypeError, ValueError):
        pass
if batch_size_large is not None:
    try:
        FileDiffSummarizer.BATCH_SIZE_LARGE = max(1, int(batch_size_large))
    except (TypeError, ValueError):
        pass
if docs_top_k is not None:
    try:
        FileDiffSummarizer.DOCS_TOP_K = max(0, int(docs_top_k))
    except (TypeError, ValueError):
        pass
if max_diff_lines_per_prompt is not None:
    try:
        FileDiffSummarizer.MAX_DIFF_LINES_PER_PROMPT = max(1, int(max_diff_lines_per_prompt))
    except (TypeError, ValueError):
        pass
FileDiffSummarizer.RANKING_CONFIG = ranking_config

# === Load environment variables (secrets only) ===

env_provider = os.getenv("LLM_PROVIDER")
PROVIDER = (env_provider or llm_config.get("provider") or "openai").lower()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_API_KEY = os.getenv("LLM_API_KEY")  # optional for local providers
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Only require a key for cloud providers
if PROVIDER == "mistral" and not MISTRAL_API_KEY:
    raise Exception("Missing MISTRAL_API_KEY in .env file.")
if PROVIDER == "openai" and not OPENAI_API_KEY:
    raise Exception("Missing OPENAI_API_KEY in .env file.")
if PROVIDER == "deepseek" and not DEEPSEEK_API_KEY:
    raise Exception("Missing DEEPSEEK_API_KEY in .env file.")
if PROVIDER == "gemini" and not GEMINI_API_KEY:
    raise Exception("Missing GEMINI_API_KEY in .env file.")
# Local/other providers (e.g., llama) do not require a cloud key.

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Generate PR description(s) using cached knowledge graph data.")
parser.add_argument("--repo_name", type=str, help="GitHub repository in 'owner/repo' format. If omitted, all PRs in the graph are processed.")
parser.add_argument("--pr", type=int, help="Pull request number to process. Requires --repo_name.")
parser.add_argument("--limit", type=int, default=None, help="Limit how many PRs are processed when running over the full graph.")
parser.add_argument(
    "--randomize",
    action="store_true",
    help="Randomize PR selection before applying --limit (only when processing all PRs).",
)
parser.add_argument(
    "--graph_path",
    type=Path,
    default=ROOT_DIR / "results" / "knowledge_graph" / f"graph-{dataset_name}.json",
    help="Path to the knowledge graph JSON."
)
args = parser.parse_args()

repo_name = args.repo_name
pr_number = args.pr
graph_path: Path = args.graph_path

if not graph_path.exists():
    raise FileNotFoundError(f"Knowledge graph file not found: {graph_path}")

if (repo_name is None) != (pr_number is None):
    raise ValueError("Provide both --repo_name and --pr, or omit both to process every PR in the graph.")

llm_key = None
if PROVIDER == "openai":
    llm_key = OPENAI_API_KEY
elif PROVIDER == "mistral":
    llm_key = MISTRAL_API_KEY
elif PROVIDER == "llama":
    llm_key = LLAMA_API_KEY or "ollama"
elif PROVIDER == "deepseek":
    llm_key = DEEPSEEK_API_KEY
elif PROVIDER == "gemini":
    llm_key = GEMINI_API_KEY

# === Load PR context from knowledge graph ===
graph_reader = KnowledgeGraphReader(graph_path)

if repo_name is not None:
    targets = [(repo_name, pr_number)]
else:
    targets = graph_reader.list_pull_requests()
    if args.randomize:
        random.shuffle(targets)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be a positive integer.")
        targets = targets[: args.limit]
    if not targets:
        raise RuntimeError("No pull requests found in the knowledge graph.")

# === Run Orchestrator ===
orchestrator = PRDescriptionOrchestrator(
    github_token=os.getenv("GITHUB_CLASSIC_TOKEN") or os.getenv("GITHUB_TOKEN"),
    llm_api_key=llm_key,
    enable_data_collection=False,
    llm_provider=PROVIDER,
    llm_settings=llm_config,
    cmg_config=cmg_config,
    ranking_config=ranking_config,
)

def clean_markdown(text: str) -> str:
    # Remove leading/trailing ```markdown or ```
    text = re.sub(r"^```(?:\w+)?\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())

    # Remove bold/italics/bullets/headings
    lines = text.splitlines()
    clean = []
    for line in lines:
        line = re.sub(r"^(\s*[-*•+]\s*)", "", line)
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        clean.append(line.strip())

    return "\n".join(clean).strip()


def build_file_list(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    file_list: List[Dict[str, Any]] = []
    for f in files or []:
        filename = f.get("filename") or ""
        if not filename:
            continue
        file_list.append(
            {
                "filename": filename,
                "status": f.get("status"),
                "additions": f.get("additions"),
                "deletions": f.get("deletions"),
                "is_docs": _is_docs_file(filename),
                "has_patch": bool((f.get("patch") or "").strip()),
            }
        )
    return file_list

# === Save to results/pr-description/<provider>/descriptions-*.json ===
results_dir = ROOT_DIR / "results" / "pr-description" / PROVIDER
results_dir.mkdir(parents=True, exist_ok=True)

def _next_results_file(dir_path: Path, mode_slugs: List[str], dataset_label: str) -> Path:
    if not mode_slugs:
        safe_slug = "modes"
    else:
        safe_slug = "modes-" + "-".join(mode_slugs)
    safe_slug = re.sub(r"[^a-zA-Z0-9]+", "-", safe_slug.lower()).strip("-") or "modes"
    safe_dataset = re.sub(r"[^a-zA-Z0-9]+", "-", dataset_label.lower()).strip("-") or "dataset"
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    return dir_path / f"descriptions-{safe_dataset}-{safe_slug}-{timestamp}.json"

llm_error_log = ROOT_DIR / "logs" / "pr-descriptions" / "errors.log"

combined_results: List[Dict[str, Any]] = []
mode_slugs = [m.get("slug", m.get("name", "")) for m in generation_modes]
combined_result_file = _next_results_file(results_dir, mode_slugs, dataset_name)
with combined_result_file.open("w", encoding="utf-8") as handle:
    json.dump([], handle)

for idx, (target_repo, target_pr) in enumerate(targets, start=1):
    print("\n" + "=" * 80)
    print(f"[MAIN] ({idx}/{len(targets)}) Repo: {target_repo} | PR #{target_pr}")
    print("=" * 80 + "\n")
    try:
        pr_context = graph_reader.get_pr_context(target_repo, target_pr)
    except Exception as exc:
        msg = f"[ERROR] Failed to load context for {target_repo}#{target_pr}: {exc}"
        print(msg)
        _log_error(llm_error_log, msg)
        continue

    original_description = pr_context.get("pr_metadata", {}).get("body") or "[Original description unavailable]"

    per_pr_record = {
        "repo_name": target_repo,
        "pr_number": target_pr,
        "original_description": original_description.strip(),
        "modes": [],
    }

    ignored_files: List[Dict[str, Any]] = []
    for f in (pr_context.get("files") or []):
        filename = f.get("filename") or ""
        if not filename:
            continue
        patch = (f.get("patch") or "").strip()
        if is_noise_path(filename):
            ignored_files.append({"filename": filename, "reason": "noise_excluded"})
            continue
        if not patch or patch.lower() == "[patch not available]":
            ignored_files.append({"filename": filename, "reason": "patch_missing"})
            continue

    if ignored_files:
        counts: Dict[str, int] = {}
        for row in ignored_files:
            reason = row.get("reason") or "unknown"
            counts[reason] = counts.get(reason, 0) + 1
        print("[MAIN] Ignored files summary:", counts)

    file_weights = (ranking_config.get("file") or {}).get("weights") or {}
    commit_weights = (ranking_config.get("commit") or {}).get("weights") or {}
    file_scores = compute_file_scores(pr_context.get("files") or [], file_weights)
    commit_ranked = rank_commits(pr_context.get("commits") or [], file_scores, commit_weights)
    commit_scores = {sha: score for sha, score in commit_ranked}

    cached_file_summaries = None
    cached_judge_summaries = None

    for mode in generation_modes:
        mode_name = mode["name"]
        slug = mode["slug"]
        options = {
            "use_cmg": mode["use_cmg"],
            "include_file_summaries": mode["include_file_summaries"],
            "include_commits": bool(mode.get("include_commits", True)),
        }

        print("-" * 80)
        print(f"[MAIN][MODE] {mode_name} ({slug})")
        print(f"[MAIN][MODE] options={options}")
        print("-" * 80 + "\n")

        precomputed_file_summaries = None
        if options["include_file_summaries"] and cached_file_summaries:
            precomputed_file_summaries = cached_file_summaries

        start_mode = time.monotonic()
        try:
            outputs = orchestrator.generate_from_context(
                pr_context,
                repo_name=target_repo,
                pr_number=target_pr,
                generation_options=options,
                precomputed_file_summaries=precomputed_file_summaries,
            )
        except Exception as exc:
            msg = f"[ERROR] LLM generation failed for {target_repo}#{target_pr} (mode={mode_name}): {exc}"
            print(msg)
            _log_error(llm_error_log, msg)
            continue
        duration = time.monotonic() - start_mode
        print(f"[MAIN][MODE {mode_name}] Duration: {duration:.2f}s\n")

        description_raw = outputs.get("description", "")
        rewritten_commits = outputs.get("rewritten_commits", [])
        commit_decisions = outputs.get("commit_decisions", [])
        file_summaries = outputs.get("file_summaries", [])
        if options["include_file_summaries"] and file_summaries:
            cached_file_summaries = file_summaries
        if cached_judge_summaries is None and cached_file_summaries:
            cached_judge_summaries = cached_file_summaries

        for fs in file_summaries or []:
            filename = fs.get("filename")
            if filename in file_scores:
                fs["score"] = round(file_scores[filename], 4)

        cleaned_description = clean_markdown(description_raw)

        summary_map = {fs.get("filename"): fs.get("summary") for fs in (file_summaries or [])}

        mode_record = {
            "generation_mode": mode_name,
            "generation_slug": slug,
            "generation_options": options,
            "duration_seconds": round(duration, 2),
            "generated_description": cleaned_description,
            "rewritten_commits": rewritten_commits,
            "commit_decisions": commit_decisions,
            "file_summaries": file_summaries,
            "file_scores": {k: round(v, 4) for k, v in file_scores.items()},
            "commit_scores": {k: round(v, 4) for k, v in commit_scores.items()},
            "ignored_files": ignored_files,
            "ignored_files_summary": {
                "noise_excluded": sum(1 for row in ignored_files if row.get("reason") == "noise_excluded"),
                "patch_missing": sum(1 for row in ignored_files if row.get("reason") == "patch_missing"),
            },
            "commit_details": [
                {
                    "sha": c.get("sha"),
                    "original_message": c.get("message"),
                    "cmg_rewritten_message": c.get("cmg_rewritten_message"),
                    "score": round(commit_scores.get(c.get("sha"), 0.0), 4),
                    "patches": c.get("patches") or [],
                }
                for c in (pr_context.get("commits") or [])
            ],
            "file_diffs": [
                {
                    "filename": f.get("filename"),
                    "summary": summary_map.get(f.get("filename")),
                    "status": f.get("status"),
                    "additions": f.get("additions"),
                    "deletions": f.get("deletions"),
                    "changes": f.get("changes"),
                    "score": round(file_scores.get(f.get("filename"), 0.0), 4),
                    "patch": f.get("patch"),
                    "is_docs": _is_docs_file(f.get("filename") or ""),
                }
                for f in (pr_context.get("files") or [])
            ],
        }

        commits_for_prompt = pr_context.get("commits") or []
        commit_cfg = (ranking_config.get("commit") or {})
        include_all_if_leq = int(commit_cfg.get("include_all_if_commit_count_leq") or 0)
        top_k = int(commit_cfg.get("top_k_large") or 0)
        if top_k and (include_all_if_leq == 0 or len(commits_for_prompt) > include_all_if_leq):
            file_weights = (ranking_config.get("file") or {}).get("weights") or {}
            file_scores = compute_file_scores(pr_context.get("files") or [], file_weights)
            ranked = rank_commits(commits_for_prompt, file_scores, commit_cfg.get("weights") or {})
            keep_shas = {sha for sha, _ in ranked[:top_k]}
            commits_for_prompt = [c for c in commits_for_prompt if c.get("sha") in keep_shas]

        commit_payload = CommitMessageRewriter.build_payload(
            commits_for_prompt,
            use_cmg_commits=bool(options.get("use_cmg")),
            max_lines_per_patch=None,
        )
        commit_payload = CommitMessageRewriter.trim_payload_by_tokens(
            commit_payload,
            int(commit_payload_config.get("max_tokens_per_prompt") or 0),
        )
        file_list = build_file_list(pr_context.get("files") or [])
        evidence_summaries = cached_judge_summaries or []
        diff_payload = FileDiffSummarizer.build_payload(
            pr_context.get("files") or [],
            max_lines_per_patch=None,
        )
        diff_map = {item.get("filename"): item for item in diff_payload}
        file_summary_diffs: List[Dict[str, Any]] = []
        for summary in evidence_summaries:
            filename = summary.get("filename")
            if not filename:
                continue
            diff = diff_map.get(filename)
            if not diff:
                continue
            file_summary_diffs.append(
                {
                    "filename": filename,
                    "summary": summary.get("summary"),
                    "status": diff.get("status"),
                    "additions": diff.get("additions"),
                    "deletions": diff.get("deletions"),
                    "diff_excerpt": diff.get("diff_excerpt"),
                }
            )
        mode_record["judge_evidence"] = {
            "commit_payload": commit_payload,
            "file_summary_diffs": file_summary_diffs,
            "file_list": file_list,
            "file_list_meta": {
                "total_files": len(pr_context.get("files") or []),
                "returned_files": len(file_list),
                "truncated": False,
            },
            "file_summaries": evidence_summaries,
        }

        per_pr_record["modes"].append(mode_record)

        print(f"\n[MAIN][MODE {mode_name}] Final Pull Request Description:\n")
        print(cleaned_description)
        print("\n")
        print(f"[MAIN][MODE {mode_name}] Commit Messages:\n")
        for decision in commit_decisions or []:
            print(f"SHA: {decision.get('sha')}")
            print(f"Original: {decision.get('original')}")
            if decision.get("final") and decision.get("final") != decision.get("original"):
                print(f"Rewritten: {decision.get('final')}")
            print(f"Status: {decision.get('status')} | Reason: {decision.get('reason')}")
            print("")

        if file_summaries:
            docs = [fs for fs in file_summaries if fs.get("is_docs")]
            non_docs = [fs for fs in file_summaries if not fs.get("is_docs")]
            print(f"[MAIN][MODE {mode_name}] File Summaries:\n")
            if non_docs:
                print("Non-doc files:")
                for fs in non_docs:
                    print(f"{fs.get('filename')}: {fs.get('summary')}")
                print("")
            if docs:
                print("Documentation files:")
                for fs in docs:
                    print(f"{fs.get('filename')}: {fs.get('summary')}")
            print("")

        print(f"[MAIN][MODE {mode_name}] File Diffs:\n")
        docs = [f for f in mode_record["file_diffs"] if f.get("is_docs")]
        non_docs = [f for f in mode_record["file_diffs"] if not f.get("is_docs")]
        if non_docs:
            print("Non-doc files:")
        for f in non_docs:
            print(f"File: {f.get('filename')}")
            if f.get("summary"):
                print(f"Summary: {f.get('summary')}")
            print(f"Status: {f.get('status')} | +{f.get('additions')} -{f.get('deletions')} ({f.get('changes')})")
            print("")
        if docs:
            print("Documentation files:")
        for f in docs:
            print(f"File: {f.get('filename')}")
            if f.get("summary"):
                print(f"Summary: {f.get('summary')}")
            print(f"Status: {f.get('status')} | +{f.get('additions')} -{f.get('deletions')} ({f.get('changes')})")
            print("")

    combined_results.append(per_pr_record)
    with combined_result_file.open("w", encoding="utf-8") as handle:
        json.dump(combined_results, handle, indent=2)

print("\n[MAIN] Generation complete.")
print(f"    Saved {len(combined_results)} PR records to {combined_result_file.absolute()}")
