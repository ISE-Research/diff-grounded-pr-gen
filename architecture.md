# Final Codebase Architecture

## Purpose of This Document

This file is intended to be a durable technical reference for `final-codebase/`, not a quick-start guide and not a directory checklist. The goal is to capture:

- how the full pipeline works end to end
- what each major component does
- what data enters and leaves each stage
- how that data is transformed
- which heuristics materially affect behavior
- which artifacts in the repository are primary outputs versus incidental traces
- what results already exist in the tree and how they relate to the pipeline

If you return to this repository later, this document should let you reconstruct the system without rereading the entire codebase first.

## One-Sentence System Description

`final-codebase/` is a staged research pipeline that starts from a dataset of GitHub pull requests, collects and caches rich PR context in a knowledge graph, generates multiple PR-description variants under controlled ablations, evaluates those variants with an LLM judge and human-study tooling, and produces final summary tables and plots for reporting.

## The Central Architectural Idea

The single most important design decision in this project is that GitHub collection is separated from later experimentation by a serialized knowledge graph.

The pipeline is therefore not:

```text
dataset -> call GitHub -> call LLM -> done
```

It is:

```text
dataset
  -> GitHub collection
  -> structured graph cache
  -> generation experiments
  -> evaluation experiments
  -> frozen reporting inputs
  -> final summary artifacts
```

This design matters because:

- GitHub collection is expensive and rate-limit sensitive.
- Generation and judge runs need to be repeatable.
- Multiple ablation modes should run over identical cached context.
- Evaluation should not silently depend on live repository state changing over time.

The knowledge graph is therefore the first major system boundary. `final-results/data/` is the second major system boundary. Everything else hangs off those two artifacts.

## High-Level Pipeline Map

The practical pipeline is:

```text
raw/source dataset CSV
  -> normalized PR target CSV
  -> PR context collection from GitHub
  -> knowledge graph JSON
  -> per-PR multi-mode description generation JSON
  -> per-mode judged evaluation JSON
  -> frozen result JSONs for reporting
  -> summary tables and plots
```

The primary directories correspond to these phases:

- `config/`: runtime configuration
- `data/`: datasets and dataset parsing helper
- `data-collection/`: GitHub collection and graph building
- `description-generation/`: main PR-description generation pipeline
- `judge/`: LLM-based evaluator and survey export tooling
- `results/`: run outputs
- `final-results/`: frozen evaluation inputs and final reporting outputs

Additional root-level files that are part of the runnable repository state:

- `README.md`: concise user-facing summary of the same pipeline
- `requirements.txt`: dependency list for the Python environment
- `run_pipeline.sh`: convenience shell runner for generation + judge
- `.env`: local secrets and runtime environment configuration

## Repository State That Matters

The repository contains both code and already-produced artifacts. For understanding the project, the most important persistent assets are:

1. `config/pipeline.yaml`
2. `results/knowledge_graph/*.json`
3. `results/pr-description/**/*.json`
4. `results/judge/**/*.json`
5. `final-results/data/description-data/*.json`
6. `final-results/eval/tables/*`
7. `final-results/eval/plots/*`

Logs exist, but they are not conceptually central. They matter for debugging failed runs, not for understanding the architecture or the research outputs.

## System Layers

The system has six functional layers.

### Layer 1: Configuration

Defines:

- active dataset
- provider and model selection
- commit-message-generation settings
- generation ablation modes
- file and commit ranking heuristics
- judge evidence and scoring parameters

Files:

- `config/pipeline.yaml`
- `config/loader.py`

### Layer 2: Dataset normalization

Defines which PRs are in scope and transforms source dataset formats into the normalized `(repo_name, pr_number)` format used everywhere else.

Files:

- `data/*.csv`
- `data/get_ids.py`

### Layer 3: GitHub collection and graph construction

Fetches PR metadata, commits, patches, files, and linked issues from GitHub, then serializes the result into a graph.

Files:

- `data-collection/build_knowledge_graph.py`
- `data-collection/knowledge_graph/graph_builder.py`
- `description-generation/components/pr_data_collector.py`
- `description-generation/wrappers/github_wrapper.py`

Also present:

- `data-collection/knowledge_graph/__init__.py`, which exposes the graph builder and reader as the package surface

### Layer 4: Description generation

Reads PR context from the graph, applies deterministic evidence selection, optionally rewrites commit messages, optionally summarizes file diffs, and generates PR descriptions under multiple modes.

Files:

- `description-generation/main.py`
- `description-generation/orchestrator/pr_orchestrator.py`
- `description-generation/components/*`
- `description-generation/wrappers/*/llm_client.py`

Supporting helper also present:

- `description-generation/components/anchor_terms.py`

### Layer 5: Automated evaluation

Compares generated descriptions against original PR descriptions and records scored judgments with detailed breakdowns.

Files:

- `judge/judge.py`
- `judge/survey_from_judge.py`

### Layer 6: Reporting and frozen analysis

Consumes selected judge outputs and human-study data to produce the tables and plots used for reporting.

Files:

- `final-results/data/*`
- `final-results/scripts/*`
- `final-results/eval/*`

Also present:

- `final-results/README.md`, which documents the reporting layout and notes that `parsed` is reported as `Trudeau`

## Configuration Layer in Detail

### `config/pipeline.yaml`

This file is the authoritative runtime configuration for the whole pipeline.

The root `README.md` is consistent with this architecture and presents the same four major runnable stages:

1. build the knowledge graph
2. generate PR descriptions
3. run the judge
4. run final analysis

The README is a compact operational summary; this document is the deeper architectural reference.

Current top-level sections:

- `llm`
- `dataset`
- `cmg`
- `generation_modes`
- `active_generation_modes`
- `file_summaries`
- `ranking`
- `judge`
- `commit_payload`

#### `llm`

Controls the provider wrapper and model used in generation, and also supplies defaults for judging.

Current configured values:

- `provider: openai`
- `base_url: null`
- `model: gpt-5-mini-2025-08-07`
- `temperature: 0.2`
- `log_prompts: true`

Effect on runtime:

- determines which wrapper class `PRDescriptionOrchestrator` resolves
- determines what API key is required
- determines generation prompt stochasticity

#### `dataset`

Controls which dataset is active.

Current configured value:

- `name: parsed`

Derived behavior:

- if only `name` is set, loader infers `data/<name>.csv`
- graph path defaults to `results/knowledge_graph/graph-<name>.json`

#### `cmg`

Controls commit-message generation and rewriting.

Current relevant values:

- enabled: `true`
- demo scope: `global`
- `k: 16`
- `batch_enabled: false`
- `max_chunk_tokens: 16000`
- `batch_demo_k: 2`
- semantic model: `sentence-transformers/all-MiniLM-L6-v2`
- QA settings:
  - `use_sem: true`
  - `score_threshold: 0.8`
  - `good_threshold: 0.65`
  - `min_improve: 0`
  - `llm_judge: false`
  - `pairwise: false`

Architectural effect:

- enables or disables the CMG subsystem entirely
- determines how demos are retrieved from the graph
- determines how strict candidate acceptance is intended to be

Important implementation note:

The config suggests strong threshold-based gating, but the current `CmgQuality.accept()` implementation effectively accepts any candidate whose score is strictly better than the original. The config values still shape some logic, but the final acceptance behavior is looser than the names imply.

#### `generation_modes`

Defines the experiment structure.

Current defined modes:

- `test_mode`
- `raw`
- `cmg_only`
- `file_summaries_only`
- `full`

Current active modes:

- `raw`
- `cmg_only`
- `file_summaries_only`
- `full`

The modes are ablations over:

- whether CMG-rewritten commit messages are used
- whether file summaries are included
- whether commits are included

This mode layer is the core of the experimental design.

#### `file_summaries`

Controls file-summary batching and prompt shaping.

Current values:

- `batch_size_small: 2`
- `batch_size_large: 3`
- `docs_top_k: 5`
- `max_diff_lines_per_prompt: 350`

Architectural effect:

- shapes how much file evidence reaches the summarizer
- constrains prompt size during file-summary generation
- determines how many documentation files can survive selection

#### `ranking`

Contains separate configurations for file ranking and commit ranking.

File ranking controls:

- whether all files are included for small PRs
- how many top files are included for larger PRs
- weights for path keywords, API impact, tests, CI, size, added/deleted status, and noise

Commit ranking controls:

- whether all commits are included for small PRs
- how many top commits are included for larger PRs
- weights for impact, intent, linked issues, verb-start, CMG quality, identifier overlap, and short-message penalty

These values matter because they control which evidence survives into prompts.

#### `judge`

Current values:

- `temperature: 0`
- `evidence.max_file_list_items: 200`

Architectural effect:

- keeps judge outputs deterministic or near-deterministic
- caps one type of evidence expansion

#### `commit_payload`

Current value:

- `max_tokens_per_prompt: 0`

Interpretation:

- `0` means effectively unlimited trimming at the commit-payload stage

### `config/loader.py`

`load_pipeline_config()` is the normalization layer used across collection, generation, and judge.

Important behaviors:

- deep-merges YAML over defaults
- infers dataset CSV path from dataset name if missing
- infers dataset name from CSV path if missing
- rewrites default CMG graph path to `results/knowledge_graph/graph-<dataset>.json`
- normalizes generation modes by assigning slugs
- filters to `active_generation_modes` when present

This file matters because it keeps the system coherent. If one stage appears to be using the wrong dataset, wrong graph path, or wrong mode set, this loader is one of the first places to inspect.

### `requirements.txt`

The dependency file is part of the architecture because it shows what subsystems the repository actually depends on.

Current dependencies map cleanly onto the architecture:

- `python-dotenv`: environment loading
- `PyYAML`: config loading
- `pandas`: dataset parsing utilities
- `PyGithub`: GitHub API access
- `openai`, `mistralai`, `google-genai`, `google-generativeai`: provider wrappers
- `networkx`: graph storage and serialization
- `numpy`: numeric helpers for semantic similarity
- `sentence-transformers`: semantic retrieval/quality scoring
- `datasets`: dataset utilities in the experimentation stack

## Dataset Layer in Detail

### Purpose

The dataset layer defines the PR universe that the pipeline processes.

### Files

Relevant files currently in `data/`:

- `aidev.csv`
- `done.csv`
- `parsed.csv`
- `parsed-1.csv`
- `test.pr_commits_20_400_100_0.5_nltk.csv`
- `train.pr_commits_20_400_100_0.5_nltk.csv`
- `valid.pr_commits_20_400_100_0.5_nltk.csv`

The active configuration points at the `parsed` dataset by default.

### `data/get_ids.py`

This helper exists because some source datasets encode PR identity in a single `id` field rather than normalized columns.

Behavior:

1. read column `id`
2. parse it into:
   - repository full name
   - PR number
3. de-duplicate while preserving order
4. optionally truncate by `--limit`
5. write `repo_name,pr_number` rows

Expected input style:

- strings like `owner_repo_123`

Output style:

```csv
repo_name,pr_number
owner/repo,123
```

This step is purely deterministic.

## Collection and Knowledge-Graph Layer in Detail

### Collection entrypoint: `data-collection/build_knowledge_graph.py`

This is the batch ingestion script that transforms dataset rows into a graph artifact.

Main startup flow:

1. load `.env`
2. load pipeline config
3. resolve dataset CSV and dataset name
4. resolve LLM provider name from config or environment
5. require GitHub token
6. parse CLI arguments
7. read PR targets
8. create `PRDescriptionOrchestrator` with:
   - data collection enabled
   - LLM components disabled
9. create `KnowledgeGraphBuilder`
10. iterate over PR targets
11. collect PR context with retry
12. ingest into graph
13. save graph after each success

CLI flags:

- `--input`
- `--output`
- `--limit`
- `--randomize`

Important operational behavior:

- graph is persisted incrementally, not only at the end
- rate-limit-like errors trigger exponential backoff
- failures are logged but do not abort the full run

### `data-collection/knowledge_graph/__init__.py`

This file is small but accurate to the architecture: it defines the package-level API by exporting:

- `KnowledgeGraphBuilder`
- `KnowledgeGraphReader`

That matches how the graph layer is imported by the rest of the system.

### GitHub collection path

The call path during graph build is:

```text
build_knowledge_graph.py
  -> PRDescriptionOrchestrator.collect_pr_context()
    -> PullRequestDataCollector.collect()
      -> GitHubWrapper methods
```

### `PullRequestDataCollector.collect()`

This is the canonical PR-context builder. It is responsible for assembling all raw data needed later.

Its steps are:

1. fetch PR metadata
2. extract source branch name
3. fetch commits and commit patches
4. fetch file-level diffs
5. extract linked issues
6. fetch repo metadata
7. return a single structured dictionary

This method is effectively the source-of-truth schema constructor for the whole project.

### `GitHubWrapper`

This wrapper isolates all GitHub API access.

#### `get_pull_request(repo_name, pr_number)`

Returns the live PR object.

#### `get_branch_name(pr)`

Returns `pr.head.ref`.

#### `get_repo_metadata(repo_name)`

Collects:

- repo name
- full name
- description
- primary language
- topics
- owner
- license
- fork flag
- stargazer count
- fork count

#### `get_pull_commits(repo_name, pr_number)`

For each commit in the PR, collects:

- SHA
- raw commit message
- author name/email/login/avatar
- timestamp
- files touched
- per-file patch content
- lightweight heuristics:
  - `is_short`
  - `starts_with_verb`

Important detail:

- each full commit is fetched separately so that file patches can be attached
- a commit’s `patches` field is a list of `{filename, patch}` dictionaries

#### `get_pull_files(repo_name, pr_number)`

Collects PR-level changed files with:

- filename
- status
- additions
- deletions
- changes
- patch or `[Patch not available]`

#### `get_linked_issues(pr)`

This method is more involved than the name suggests.

It combines:

- regex scanning for issue-closing keywords such as close/fix/resolve
- parsing of issue references:
  - full GitHub issue URLs
  - `owner/repo#123`
  - `#123`
- GraphQL queries for:
  - `closingIssuesReferences`
  - `referencedIssues`

Important caveat:

- the implementation currently scans commit messages for keyword references, not the PR body itself, even though some print statements imply PR-body scanning

### Canonical PR context schema

The returned PR context is:

```python
{
  "pr_number": int,
  "repo": str,
  "branch_name": str,
  "pr_metadata": {...},
  "linked_issues": [...],
  "commits": [...],
  "files": [...],
  "repo_metadata": {...},
}
```

#### `pr_metadata`

Collected fields include:

- `id`
- `number`
- `title`
- `body`
- `state`
- `is_draft`
- `created_at`
- `updated_at`
- `closed_at`
- `merged_at`
- `merge_commit_sha`
- `html_url`
- `author_login`
- `author_name`
- `author_avatar_url`
- `base_branch`
- `base_repo`
- `head_branch`
- `head_repo`
- `additions`
- `deletions`
- `changed_files`
- `labels`

#### `linked_issues`

Each issue entry may contain:

- `number`
- `repo`
- `title`
- `body`
- `state`
- `url`
- `keyword_used`
- `source`

#### `commits`

Each commit entry initially contains:

- `sha`
- `message`
- `author`
- `author_email`
- `author_login`
- `author_avatar_url`
- `timestamp`
- `files_touched`
- `patches`
- `is_short`
- `starts_with_verb`

#### `files`

Each file entry contains:

- `filename`
- `status`
- `additions`
- `deletions`
- `changes`
- `patch`

#### `repo_metadata`

Contains:

- `name`
- `full_name`
- `description`
- `language`
- `topics`
- `owner`
- `license`
- `is_fork`
- `stargazers_count`
- `forks_count`

### Knowledge graph construction: `graph_builder.py`

The graph is a `networkx.MultiDiGraph`.

Node types:

- `Repository`
- `PullRequest`
- `Issue`
- `Commit`
- `File`

Edge types:

- `REPO_HAS_PR`
- `PR_LINKS_ISSUE`
- `PR_CONTAINS_COMMIT`
- `PR_MODIFIES_FILE`

Important node IDs:

- repository: `repo:<full_name>`
- PR: `pr:<repo>#<number>`
- issue: `issue:<repo>#<number>`
- commit: `commit:<sha>`
- file: `file:<repo>:<path>`

Important behaviors:

- repository and PR nodes are upserted
- duplicate edges with identical label/attrs are suppressed
- PR body may be stored on the PR node
- file patch content is stored on the PR-to-file edge

### Commit quality annotations in the graph

Before commit nodes are written, `compute_commit_quality_annotations()` is run.

This computes deterministic features such as:

- whether the message is already “good”
- whether it likely needs rewrite
- whether it is merge/revert
- whether it starts with a verb
- message length and token count
- issue references
- tests indicators
- diff token count
- semantic similarity estimate
- identifier overlap
- overall quality score

This is a major architectural detail because the graph is not a raw dump of GitHub data. It already contains derived features that later ranking and CMG logic rely on.

### Graph read layer: `graph_reader.py`

`KnowledgeGraphReader` reconstructs PR context from graph JSON.

Public methods:

- `get_pr_context(repo_name, pr_number)`
- `list_pull_requests()`

Important behavior:

- if the expected PR node ID is not found, it falls back to scanning nodes by attributes
- commits are returned with guaranteed `patches` and `files_touched` keys
- missing file patches are normalized to `[Patch not available]`

This reader reconstitutes the same conceptual PR context that collection originally produced, which is what allows later stages to run from the graph rather than GitHub.

## Description-Generation Layer in Detail

### Entrypoint: `description-generation/main.py`

This script is the main experiment runner.

Its startup behavior is:

1. load `.env`
2. load config
3. resolve provider and validate API key requirements
4. apply file-summary config values to `FileDiffSummarizer` class variables
5. resolve graph path
6. create `KnowledgeGraphReader`
7. choose targets:
   - one PR via CLI
   - or all PRs in graph
8. instantiate `PRDescriptionOrchestrator` with:
   - data collection disabled
   - LLM components enabled
9. iterate over PRs
10. iterate over modes per PR
11. write one combined results JSON incrementally

CLI flags:

- `--repo_name`
- `--pr`
- `--limit`
- `--randomize`
- `--graph_path`

Important runtime behaviors:

- either both `--repo_name` and `--pr` must be supplied or neither
- the graph file must already exist
- file scores and commit scores are computed before mode iteration
- file summaries can be cached across modes for the same PR

### Output structure from `main.py`

Each output file is:

```python
[
  {
    "repo_name": str,
    "pr_number": int,
    "original_description": str,
    "modes": [mode_record, ...]
  },
  ...
]
```

Each `mode_record` includes:

- `generation_mode`
- `generation_slug`
- `generation_options`
- `duration_seconds`
- `generated_description`
- `rewritten_commits`
- `commit_decisions`
- `file_summaries`
- `file_scores`
- `commit_scores`
- `ignored_files`
- `ignored_files_summary`
- `commit_details`
- `file_diffs`
- `judge_evidence`

This is not just a text-generation output. It is an experiment record with attached evidence.

### `PRDescriptionOrchestrator`

This class coordinates generation-time components.

It can be initialized in two different roles:

- collection mode:
  - GitHub enabled
  - LLM disabled
- generation mode:
  - GitHub disabled
  - LLM enabled

Main responsibilities:

- initialize correct provider wrapper
- initialize `PRDescriptionGenerator`
- initialize CMG rewriter when enabled
- collect live PR context when requested
- generate descriptions from cached context

### Provider wrapper resolution

The orchestrator resolves wrappers using provider names:

- `openai`
- `mistral`
- `llama`
- `deepseek`
- `gemini`

This resolution is centralized in `_resolve_llm_client()`.

### Generation mode execution inside orchestrator

For each PR/mode:

1. decide whether CMG should run
2. decide whether file summaries should be included
3. decide whether commits should be included
4. optionally run CMG on selected commits
5. call `PRDescriptionGenerator.generate_outputs()`
6. merge CMG decisions into outputs

### CMG inside orchestrator

`_run_cmg_on_commits()` performs an important pre-generation transformation.

For large PRs:

- files are ranked
- commits are ranked based on file impact and intent signals
- only top commits may be eligible for rewrite

The rewriter returns decisions that are merged back into `pr_context["commits"]`, adding fields such as:

- `cmg_status`
- `cmg_reason`
- `cmg_candidate_message`
- `cmg_final_message`
- `cmg_rewritten_message`

The PR context is then marked with `_cmg_done = True` so multiple modes can reuse the same rewrite results.

### `PRDescriptionGenerator`

This component owns the final PR-description generation call.

It performs four main tasks:

1. generate or reuse file summaries
2. build the generation payload
3. build the system and user prompts
4. parse the structured JSON response

#### Prompt payload contents

Depending on mode, payload may include:

- linked issues
- ranked/truncated commits
- file summaries
- raw file payloads
- file summary diff excerpts

The prompts enforce:

- JSON-only output
- no use of external knowledge
- fixed markdown sections
- grounded claims only
- short bullets
- tests line only when explicitly evidenced
- linked issues only when explicitly present

This component therefore sits at the boundary between deterministic preprocessing and final model generation.

### `FileDiffSummarizer`

This component does both evidence selection and LLM summarization.

Its work can be divided into two phases.

#### Phase 1: deterministic file selection

For each changed file it determines:

- whether the file is docs
- inferred language
- whether public symbols were added/removed
- whether it looks like API-impacting code
- whether it is a test file
- whether it is a CI file
- whether the path contains risky keywords
- whether the path is noise
- an overall file score

It then selects files using policy:

- small PRs: include all non-noise non-doc files
- larger PRs: keep top-K plus always-include important files
- docs files are scored separately and capped separately

#### Phase 2: LLM summarization

Selected file payloads are batched and summarized.

Important constraints:

- each summary is one sentence
- JSON-only response format
- batch parsing fallback to single-file calls when malformed
- diff excerpts are truncated/cleaned before sending

This component matters because it largely determines what file-level evidence survives into final description generation and later judge evidence.

### `CommitMessageRewriter` in `components/commit_message_rewriter.py`

This is a utility helper, not the research CMG engine.

Its responsibilities are:

- combine patches from a commit into one diff excerpt
- choose original message or CMG-rewritten message
- build commit payloads for prompts and judging
- trim payloads by approximate token budget

It is used by both generation and judge.

### CMG subsystem: `components/cmg_commit_rewriter.py`

This is one of the most specialized parts of the system.

Its architecture has three main pieces:

1. diff rendering
2. demo retrieval from the graph
3. candidate generation and acceptance

#### Diff rendering

`_build_diff()` converts commit patches into a tagged representation using:

- `<FILE>`
- `<CTX>`
- `<ADD>`
- `<DEL>`

This tagged format is what retrieval and prompting operate on.

#### Demo retrieval: `_GraphDemoPool`

The demo pool loads the knowledge graph and extracts only “good” commit-message examples.

Each demo record stores:

- diff text
- message
- repo
- PR number
- diff token count
- files touched

Retrieval signals:

- token-overlap/Jaccard
- BM25
- optional sentence-transformer semantic similarity
- change-size bonus
- files-touched bonus

Supported scopes:

- global
- repo
- PR

Important filtering rule:

- demos from the exact same `(repo, pr)` are excluded

#### Candidate generation

Candidate generation can happen one-by-one or in batch. The current config has `batch_enabled: false`, but the subsystem supports both.

The prompt instructs the model to:

- write a single commit message
- use only diff facts
- avoid external knowledge
- mention at most 1-2 key files/symbols
- keep under about 30 words
- return JSON only

#### Candidate acceptance

Candidates are not accepted automatically.

The acceptance pipeline checks:

- whether commit was selected for rewrite
- whether commit is merge/revert
- whether patch exists
- whether original message is already good
- whether generated candidate is grounded in diff tokens
- whether strict grounding passes
- whether heuristic quality improves
- optional LLM judge approval

Output per commit includes:

- original message
- candidate message
- final chosen message
- kept/rewritten status
- reason
- patch availability

### `cmg_quality.py`

This file serves two different purposes:

1. lightweight deterministic commit-quality annotations for graph storage
2. runtime heuristic scoring for CMG acceptance

Important functions:

- `is_merge_or_revert()`
- `is_good_commit_message()`
- `needs_cmg()`
- `compute_commit_quality_annotations()`
- `CmgQuality.score()`
- `CmgQuality.accept()`

Signals used:

- vague-word detection
- imperative-like starts
- token overlap with diff
- TF-IDF cosine
- semantic similarity
- anchor-token bonuses
- identifier overlap

Architecturally, this file is a bridge between collection-time feature derivation and generation-time acceptance logic.

### `ranking.py`

This file provides deterministic selection pressure on the pipeline.

#### File ranking

Signals:

- path keywords
- test path
- CI path
- noise patterns
- API-impact regex matches
- size
- added/deleted status

Important path keywords include terms like:

- api
- auth
- security
- config
- schema
- migration
- workflow
- build
- deploy

#### Commit ranking

Signals:

- impact from touched-file scores
- issue-link presence
- verb-start
- CMG quality score
- CMG identifier overlap
- short-message penalty

This ranking logic affects:

- which commits are passed into prompts
- which commits CMG may rewrite on large PRs
- which evidence is most visible to the model

### `patch_utils.py`

Shared helpers for:

- removing git headers
- removing index lines
- removing blank lines
- truncating patch length
- combining patches into one excerpt

This matters because nearly every prompt-facing diff in the system passes through these transformations.

### `anchor_terms.py`

This helper extracts identifier-like grounding tokens from input texts using:

- identifier regexes
- path-like token regexes
- stopword and noise-token filtering
- heuristics for camelCase, underscore tokens, and numeric/code-like terms

It is not a dominant part of the current top-level execution path, but it is part of the description-generation subsystem and is now accounted for here.

### LLM provider wrappers

Provider wrappers all expose a common conceptual interface:

```python
chat(system_prompt, user_prompt, log_type, repo, pr_number) -> str
```

Wrappers exist for:

- OpenAI
- Mistral
- DeepSeek
- Gemini
- local Llama/Ollama

Common behaviors:

- prompt/response logging
- retry/backoff
- provider-specific SDK usage

Important wrapper caveat:

- wrappers return plain strings, including error sentinel strings
- downstream code often assumes a string response and only sometimes special-cases wrapper errors

The OpenAI wrapper is the most special because it explicitly detects context-too-large conditions and returns `[LLM skipped: context_too_large]`, which the judge code knows how to handle.

## Automated Judge Layer in Detail

### Entrypoint: `judge/judge.py`

This file is the main automated evaluation control script.

It consumes:

- one descriptions JSON file
- one knowledge graph JSON
- one chosen judge provider

It produces:

- a list of per-mode judged output rows

Main startup flow:

1. load `.env`
2. load config
3. resolve default judge provider
4. locate descriptions file
5. load graph
6. initialize provider-specific judge wrapper
7. optionally resume from previous judge file
8. iterate over PRs
9. iterate over modes within each PR
10. rebuild evidence payloads
11. run full judgment prompt
12. compute scores from penalties
13. write incremental judge JSON

CLI flags:

- `--descriptions_path`
- `--graph_path`
- `--provider`
- `--limit`
- `--previous`

### Judge evidence construction

The judge does not rely only on generation output text. It reconstructs evidence using the graph-backed PR context.

Evidence includes:

- commit payloads
- file-summary diffs
- file list
- linked issues
- file summaries

The existence of `judge_evidence` in generation outputs does not make that the final authority. The judge rebuilds evidence again from canonical context.

### Prompt budgeting and chunking

The judge has to handle large PRs and large evidence payloads, so it estimates prompt size and can chunk evidence.

Key behaviors:

- approximate prompt tokens from character count
- apply a configurable safety factor
- if budget is exceeded, split file evidence into chunks
- if still too large, truncate diff excerpts more aggressively
- if the wrapper returns a skip sentinel, the PR/mode can be skipped

This is an important architectural safeguard because the judge is often the largest prompt stage in the system.

### Scoring model

The judge uses a penalty-subtraction rubric.

For a description:

- start at `5.0`
- subtract correctness penalties up to `2.0`
- subtract coverage penalties up to `2.0`
- subtract clarity penalties up to `1.0`
- clamp final score into `[1.0, 5.0]`

The generated description is preferred only if its final score exceeds the original score.

### Full judgment prompt

The main current path uses a “full judgment” prompt that evaluates correctness, coverage, and clarity together but returns separate penalty arrays and penalty values.

It instructs the model to:

- use only the provided evidence
- not infer beyond evidence
- treat linked issues only as explicit motivation evidence
- require concrete tokens in summary and bullets
- penalize missing tests coverage when test evidence exists
- be more forgiving about doc/template grouping

### Judge output schema

Each row in the judge output includes:

- repo and PR identifiers
- generation mode
- original description
- generated description
- `judgment` object with:
  - original score
  - generated score
  - preference
  - original breakdown
  - generated breakdown
  - prompt step artifacts
- `rubric`
- `analysis` with primary failure reason

This is the core machine-evaluation artifact for the project.

### Survey export: `survey_from_judge.py`

This script converts judge outputs into a survey-ready export grouped by PR and mode. It preserves:

- original description metrics
- per-mode generated description metrics

This is the bridge from automated evaluation outputs into human-study packaging.

## Survey Tooling in `results/survey/`

The `results/survey/` directory is partly a results directory and partly a tooling area for human-study packaging.

Files currently present include:

- `PR-Descriptions-Survey_Template.qsf`
- `build_qualtrics_survey.py`
- `generate_survey.py`
- `generated_survey.qsf`
- timestamped `survey-*.json` exports

### `build_qualtrics_survey.py`

This is a substantive script that builds a Qualtrics-importable survey from survey JSON exports.

Its hard-coded workflow shows the intended post-judge process:

- load one AIDev survey JSON
- load one parsed/PRSummarizer survey JSON
- sample `K_EACH = 10` PRs from each
- optionally deduplicate across datasets
- optionally shuffle which mode maps to A/B/C/D/E within each PR
- clone template blocks/questions from a QSF template
- write `generated_survey.qsf`

Architecturally, this script sits after judge outputs have already been transformed into survey JSON rows.

### `generate_survey.py`

This file currently exists but is empty. It is therefore accounted for as part of the repository state, but it is not currently contributing behavior.

## Final-Results Layer in Detail

This layer is not an afterthought. It is the reporting layer and should be treated as part of the architecture.

### `final-results/data/description-data/`

This directory contains frozen machine-evaluation inputs used by reporting scripts.

Current files include:

- `descriptions-aidev-modes-raw-cmg-only-file-summaries-only-full-20260124-023308-461478-judge-20260125-010236-364060.json`
- `descriptions-parsed-modes-raw-cmg-only-file-summaries-only-full-20260124-010830-751653-judge-20260124-234250-571918.json`
- `results_aidev_c.json`
- `results_aidev_z.json`
- `results_parsed_c.json`
- `results_parsed_z.json`

Interpretation:

- some frozen inputs are direct judge outputs
- some are alternate named result snapshots
- analysis scripts treat this directory as the canonical reporting input source

The bundled `final-results/README.md` confirms this layout and records one reporting convention:

- files containing `parsed` correspond to the curated dataset from Trudeau et al. and are reported as `Trudeau` in evaluation outputs

### `final-results/data/human-data/`

Contains the human-study inputs:

- survey CSV
- researcher key mapping blinded labels back to modes/datasets

### `analyze_descriptions.py`

Aggregates judged outputs into per-dataset per-mode summary statistics.

Key behavior:

- scans all JSONs in `data/description-data`
- infers dataset from filename
- filters to PRs that have all required modes
- de-duplicates by first-seen `(repo, pr, mode)`
- synthesizes an `original` baseline row
- computes averages, medians, stdevs, and preference rates

Outputs:

- `descriptions_summary.json`
- `descriptions_summary.csv`
- `descriptions_summary.md`
- plot image/PDF

### `analyze_failure_reasons.py`

Classifies penalty strings into categories such as:

- missing key changes
- unsupported specifics
- missing test coverage
- linked issue mismatch
- format/structure
- other

Outputs:

- failure summary tables

### `analyze_lexical_metrics.py`

Computes lexical overlap metrics using original PR descriptions as references.

Metrics:

- BLEU-4
- ROUGE-1 F1
- ROUGE-2 F1
- ROUGE-L F1

Outputs:

- lexical summary tables
- delta-from-raw tables

### `analyze_human_evaluation.py`

Processes blinded survey responses and researcher-key mappings.

Produces:

- ranking summaries
- first-place counts
- pairwise comparisons
- reason-theme summaries
- human-evaluation plots

### Final output directories

`final-results/eval/tables/` currently contains:

- `descriptions_summary.{json,csv,md}`
- `failure_analysis_summary.{json,csv,md}`
- `human_evaluation_summary.{json,csv,md}`
- `lexical_metrics_summary.{json,csv,md}`
- `lexical_metrics_delta_from_raw.{json,csv,md}`

`final-results/eval/plots/` currently contains:

- `descriptions_summary.{png,pdf}`
- `human_evaluation_summary.{png,pdf}`

These are the project’s terminal deliverables as currently checked in.

At the moment, the checked-in plot outputs cover:

- description summary
- human evaluation summary

The lexical and failure analyses are currently represented in the checked-in repository through tables rather than corresponding plot files.

## Concrete Artifacts Already Present in `results/`

### Knowledge graphs

Currently present:

- `results/knowledge_graph/graph-aidev.json`
- `results/knowledge_graph/graph-parsed.json`

These indicate that at least two datasets have already been ingested into the graph layer.

### Generation outputs

Currently present under `results/pr-description/openai/` are multiple timestamped runs for both `aidev` and `parsed`.

Examples:

- `descriptions-aidev-modes-raw-cmg-only-file-summaries-only-full-20260124-023308-461478.json`
- `descriptions-parsed-modes-raw-cmg-only-file-summaries-only-full-20260124-010830-751653.json`

These are the raw generation-stage experiment outputs.

### Judge outputs

Currently present under `results/judge/openai/` are multiple judged outputs corresponding to those generation runs.

Examples:

- `descriptions-aidev-modes-raw-cmg-only-file-summaries-only-full-20260124-023308-461478-judge-20260124-224006-778897.json`
- `descriptions-parsed-modes-raw-cmg-only-file-summaries-only-full-20260122-233626-175969-judge-20260122-234104-990833.json`

These are the main machine-evaluation artifacts prior to freezing into `final-results/data/`.

### Survey artifacts

`results/survey/` contains:

- Qualtrics template/build files
- generated survey QSF
- timestamped survey JSON exports

This shows that the project includes not only automated scoring but also a human-evaluation preparation path.

### Convenience runner: `run_pipeline.sh`

This shell script is a real operational entrypoint, even though it is small.

Its behavior is:

1. move into `final-codebase/`
2. activate `pr-agent-env` if present
3. run `description-generation/main.py --limit 10 --randomize`
4. run `judge/judge.py`

It assumes that the graph already exists, so it is not a full bootstrap script. It is a quick generation-plus-judge sanity-run path.

## Data Transformation Summary

This section compresses the end-to-end data transformations into one map.

### Transformation 1: source dataset -> normalized target rows

Input:

- source CSV with an `id` column or pre-normalized `repo_name,pr_number`

Output:

- rows of `(repo_name, pr_number)`

Transformation type:

- pure parsing and de-duplication

### Transformation 2: target row -> PR context dictionary

Input:

- repository name
- PR number
- GitHub API responses

Output:

- canonical PR context dictionary

Transformation type:

- live API aggregation

### Transformation 3: PR context dictionary -> graph substructure

Input:

- PR context

Output:

- repository, PR, issue, commit, and file nodes plus relationship edges

Transformation type:

- structural serialization with derived commit features

### Transformation 4: graph -> reconstructed PR context

Input:

- graph JSON

Output:

- PR context dictionary

Transformation type:

- structural deserialization

### Transformation 5: PR context -> filtered/ranked evidence

Input:

- commits
- files
- linked issues
- ranking weights

Output:

- selected commits
- selected files
- cleaned diff excerpts
- optional file summaries
- optional rewritten commit messages

Transformation type:

- deterministic filtering + optional LLM summarization/rewriting

### Transformation 6: filtered evidence -> generated PR description

Input:

- generation prompt payload

Output:

- one PR description per mode

Transformation type:

- LLM generation

### Transformation 7: generated description + evidence -> judge output

Input:

- original description
- generated description
- rebuilt evidence payload

Output:

- scored judged row with penalty breakdowns

Transformation type:

- LLM evaluation + deterministic score calculation

### Transformation 8: judged rows -> final tables and plots

Input:

- frozen judged outputs
- human-evaluation inputs

Output:

- summary tables
- summary plots

Transformation type:

- deterministic aggregation and analysis

## Detailed File and Artifact Appendix

This appendix is intentionally more granular than the main architecture narrative. The goal is to make later re-entry cheaper by recording the practical role of each important file and artifact family.

### Root files

#### `README.md`

This is the user-facing runnable overview. It does not define behavior directly, but it is useful because it captures the intended operational sequence:

1. build the knowledge graph
2. generate descriptions
3. judge descriptions
4. run final analysis

It also documents the expected environment variables and default output locations. If the code and README ever diverge, the code is authoritative.

#### `requirements.txt`

This file gives a compact picture of the system’s external dependencies:

- data/config handling: `python-dotenv`, `PyYAML`, `pandas`
- GitHub collection: `PyGithub`
- provider clients: `openai`, `mistralai`, `google-genai`, `google-generativeai`
- graph/storage/math: `networkx`, `numpy`
- semantic retrieval/scoring: `sentence-transformers`
- dataset utilities: `datasets`

The absence of a large web framework or database dependency reinforces that the project is a script-driven experimental pipeline rather than a service.

#### `run_pipeline.sh`

This file is a thin operational wrapper. It is important because it encodes the expected “fast path” for a sanity run:

- activate local env if present
- run generation on a random sample of 10 PRs
- run judge immediately after

It notably does not build the graph. That tells you graph construction is assumed to be a prior one-time or infrequent step.

### Dataset files in `data/`

#### `parsed.csv`

This is the default active dataset inferred by current config. It is therefore the baseline dataset for normal runs unless config is changed.

#### `aidev.csv`

This is an alternate dataset with its own graph and result artifacts already present in the repository.

#### `parsed-1.csv`, `done.csv`

These exist in the tree but are not active by current config. They are still worth noting because they indicate experimentation with alternate filtered or intermediate PR target sets.

#### `train/valid/test.pr_commits_20_400_100_0.5_nltk.csv`

These appear to be source dataset partitions or precursor inputs from which normalized PR targets can be derived using `get_ids.py`.

### Knowledge-graph package files

#### `data-collection/knowledge_graph/__init__.py`

Exports the two graph-layer classes:

- `KnowledgeGraphBuilder`
- `KnowledgeGraphReader`

This makes the graph subsystem importable as a coherent package rather than as ad hoc module paths.

#### `graph_builder.py`

This file is responsible for:

- mapping a PR context dictionary into graph nodes and edges
- assigning stable graph identifiers
- enriching commit nodes with quality annotations
- serializing the graph in node-link JSON form

It is the write-side contract of the graph boundary.

#### `graph_reader.py`

This file is the read-side contract of the graph boundary. It reverses graph storage back into the PR context shape expected by downstream code.

The builder and reader together define one of the most important interfaces in the repository.

### Description-generation component files

#### `pr_data_collector.py`

This file is the collector-side schema assembler. It decides the exact in-memory structure that later stages assume.

#### `pr_description_generator.py`

This file owns:

- final PR-description payload composition
- final system/user prompt construction
- response parsing
- description/file-summary output packaging

It is the last transformation before the final generation model call.

#### `file_diff_summarizer.py`

This file is both a selector and a summarizer. It decides:

- which files survive
- how docs are treated relative to code files
- how file diffs are batched
- what diff excerpt form the LLM sees

It has high leverage over generated-description quality because it strongly controls evidence exposure.

#### `commit_message_rewriter.py`

Despite the name, this file is not doing research-grade CMG inference. It is mainly a shared payload/utility layer that:

- builds commit prompt payloads
- swaps in CMG rewrites when requested
- trims commit payloads by token budget

#### `cmg_commit_rewriter.py`

This is the actual research-grade commit-message rewriting subsystem. It is a mini-pipeline inside the main pipeline:

- render commit diff
- retrieve demos from the graph
- prompt for a candidate
- apply multiple acceptance checks
- return keep/rewrite decisions

Because it uses retrieval plus heuristics plus optional judge logic, it is one of the most complex single files in the project.

#### `cmg_quality.py`

This file is shared by both graph build and generation:

- during graph build, it adds deterministic commit-quality features
- during CMG, it scores original/candidate messages against diff evidence

It is important because it connects the graph layer to later generation behavior.

#### `ranking.py`

This is one of the most consequential deterministic files in the project. It influences:

- file selection
- commit selection
- CMG rewrite eligibility on larger PRs
- prompt evidence density

The project’s outputs can change materially if ranking logic changes, even with the same model and prompts.

#### `patch_utils.py`

This file normalizes diff text before it enters prompts. Even though it is small, it affects nearly every prompt-facing diff in the system.

#### `anchor_terms.py`

This helper is less central than ranking or file summarization, but it records the project’s grounding-oriented design style: token extraction is treated as a reusable primitive rather than being buried inside prompts.

### Wrapper files

#### `github_wrapper.py`

This is the only GitHub API abstraction in the repository. It is important because all live collection semantics are centralized here.

#### Provider wrappers

- `wrappers/openai/llm_client.py`
- `wrappers/mistral/llm_client.py`
- `wrappers/deepseek/llm_client.py`
- `wrappers/gemini/llm_client.py`
- `wrappers/llama/llm_client.py`

All expose conceptually similar `chat(...)` methods, but provider-specific behavior differs:

- OpenAI wrapper has explicit context-too-large handling
- Gemini wrapper concatenates system and user prompts differently than the OpenAI-style wrappers
- local Llama/Ollama uses an OpenAI-compatible endpoint rather than a cloud API

These wrappers are operational glue rather than research logic, but they are critical integration points.

### Judge files

#### `judge.py`

This is the largest evaluation control file in the repository. It contains:

- input-file discovery
- prompt budgeting
- evidence reconstruction
- scoring rubric logic
- incremental result writing

If the project’s machine-evaluation behavior changes, this is one of the first files to inspect.

#### `survey_from_judge.py`

This is a transformation utility from judged rows into survey-export structure. It preserves:

- original description score data
- per-mode generated score data

It effectively bridges machine evaluation and human-study packaging.

### Survey-build files under `results/survey/`

#### `build_qualtrics_survey.py`

This is not just an output artifact. It is an actual construction script that takes survey JSON exports and turns them into a randomized Qualtrics survey file.

Important design choices encoded in the script:

- fixed sample size per dataset
- optional cross-dataset deduplication
- randomized A/B/C/D/E assignment per PR
- reuse of a QSF template rather than generating a survey from scratch

#### `generate_survey.py`

This file is currently empty. It does not influence behavior today, but it is part of the repository state and may have been intended as a simpler or newer survey entrypoint.

#### `PR-Descriptions-Survey_Template.qsf`

This is a template artifact required by `build_qualtrics_survey.py`. It is part of the operational survey-build pipeline.

#### `generated_survey.qsf`

This is an output artifact produced by the survey-build step and demonstrates that the survey packaging flow has already been run.

### Final-results analysis scripts

#### `analyze_descriptions.py`

This script produces the main machine-evaluation summary. More specifically, it:

- merges all available frozen description-result JSONs
- filters to PRs that have a complete set of modes
- deduplicates repeated rows
- computes per-dataset mode statistics
- synthesizes an `original` baseline row
- writes JSON, CSV, Markdown, and plot outputs
- syncs outputs to `research-paper/final-results/` if that directory exists

Its Markdown output is one of the highest-signal entrypoints for understanding current performance.

#### `analyze_failure_reasons.py`

This script works at the penalty-string level. It converts qualitative judge penalty text into a smaller taxonomy of failure categories. That makes it the main script for understanding where generated descriptions fail rather than merely how often they win.

#### `analyze_lexical_metrics.py`

This script computes lexical overlap metrics against original PR descriptions and then also computes deltas from raw mode, optionally joining those metrics conceptually with judge summary means.

It therefore serves as a bridge between language-overlap analysis and judge-based quality analysis.

#### `analyze_human_evaluation.py`

This script is the main human-study analysis layer. It:

- parses the blinded survey CSV
- parses the researcher key
- reconstructs which mode each label corresponded to
- computes rank statistics, first-place rates, pairwise preferences, task winner rates, and qualitative theme counts
- writes tables and plots

This is the main counterpart to `analyze_descriptions.py` on the human side.

## Detailed Result-Family Appendix

### `results/knowledge_graph/*.json`

These are the reusable offline caches produced by the collection stage. They represent the canonical stored PR context for each dataset.

Current checked-in families:

- `graph-aidev.json`
- `graph-parsed.json`

### `results/pr-description/openai/descriptions-*.json`

These are direct outputs from `description-generation/main.py`.

Naming pattern:

```text
descriptions-<dataset>-modes-<mode-slugs>-<timestamp>.json
```

What they contain:

- PR-level records
- nested mode outputs
- generated descriptions
- evidence attachments
- deterministic ranking outputs

These files are raw experiment outputs, not yet judged.

### `results/judge/openai/*-judge-*.json`

These are outputs from `judge/judge.py`.

Naming pattern:

```text
<descriptions-stem>-judge-<timestamp>.json
```

What they contain:

- flattened per-mode judged rows
- original and generated descriptions
- score breakdowns
- preference
- rubric metadata

These are the main machine-evaluation outputs.

### `results/survey/survey-*.json`

These are survey-ready exports created from judge outputs, not raw Qualtrics files. They are intermediate artifacts for human-study packaging.

### `final-results/data/description-data/*.json`

These are the frozen reporting inputs. Architecturally, this directory means:

- the project distinguishes “run outputs” from “reporting inputs”
- final analysis is insulated from every new run in `results/`

### `final-results/eval/tables/*.json|*.csv|*.md`

These are the stable reporting outputs. They are effectively the published internal summaries of the project state.

The Markdown files are especially important because they are human-readable summaries of the current evaluated state.

### `final-results/eval/plots/*.png|*.pdf`

These are figure artifacts intended for reporting and likely paper integration.

## Additional Notes on What Is and Is Not Architecturally Important

Important:

- source/config/data files that define behavior
- graph caches
- generation outputs
- judge outputs
- frozen reporting inputs
- final tables and plots
- survey-building scripts and template artifacts

Secondary:

- logs
- `.DS_Store`
- `__pycache__`
- `pr-agent-env/` internals

The repository mixes these categories in one tree, but they should not be weighted equally when reading it.

## Key Architectural Strengths

- The graph boundary makes experimentation repeatable.
- Evidence selection is explicit rather than implicit.
- Generation is organized around controlled ablation modes.
- Judge outputs preserve score breakdowns rather than only scalar preferences.
- Reporting reads from frozen inputs, which protects final outputs from accidental drift.
- The repository already contains the artifacts necessary to study both pipeline behavior and final results.

## Key Architectural Weaknesses and Risks

- Most interfaces are untyped dictionaries; schema safety relies on convention.
- Important heuristics are distributed across multiple modules.
- Wrapper failures are often encoded as plain strings.
- CMG acceptance semantics are more permissive than config names imply.
- The tree mixes source code and generated state, which increases navigation complexity.
- Changes to ranking or evidence shaping can strongly alter outcomes even when prompts stay the same.

## Practical Re-Entry Reading Order

### If you want to understand how the system works

Read in this order:

1. `config/pipeline.yaml`
2. `data-collection/build_knowledge_graph.py`
3. `data-collection/knowledge_graph/graph_builder.py`
4. `data-collection/knowledge_graph/graph_reader.py`
5. `description-generation/main.py`
6. `description-generation/orchestrator/pr_orchestrator.py`
7. `description-generation/components/pr_description_generator.py`
8. `description-generation/components/file_diff_summarizer.py`
9. `description-generation/components/cmg_commit_rewriter.py`
10. `description-generation/components/ranking.py`
11. `judge/judge.py`
12. `final-results/scripts/analyze_descriptions.py`

### If you want to understand what the project has already produced

Read in this order:

1. `final-results/eval/tables/descriptions_summary.md`
2. `final-results/eval/tables/human_evaluation_summary.md`
3. `final-results/eval/tables/failure_analysis_summary.md`
4. `final-results/eval/tables/lexical_metrics_summary.md`
5. `final-results/data/description-data/*`
6. `results/judge/openai/*`
7. `results/pr-description/openai/*`
8. `results/knowledge_graph/*.json`
9. `config/pipeline.yaml`

## Final Architectural Summary

The project is best understood as a chain of persistent transformations:

- datasets choose PRs
- collection builds canonical PR context
- the graph freezes that context
- generation creates ablated descriptions from the frozen context
- the judge scores those descriptions against the original
- selected judge outputs are frozen for reporting
- reporting scripts produce the final tables and plots

The knowledge graph is the collection boundary. The descriptions JSON is the generation boundary. The judge JSON is the evaluation boundary. `final-results/data/` is the reporting boundary. The existing tables and plots are the current end products of the system.
