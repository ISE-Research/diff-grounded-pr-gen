# Installation

Estimated setup time: less than 5 minutes, excluding API account setup and
full pipeline runtime.

## Requirements

- Python 3.10 or newer
- GitHub token for the artifact-collection stage
- LLM API key for the configured provider

The default provider is configured in `config/pipeline.yaml`.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the credentials required for the run:

```bash
export GITHUB_TOKEN=...
export OPENAI_API_KEY=...
```

Alternative providers can be configured with:

```bash
export LLM_PROVIDER=openai
export MISTRAL_API_KEY=...
export DEEPSEEK_API_KEY=...
export GEMINI_API_KEY=...
```

## Smoke Test

Build a small cached PRContext:

```bash
python data-collection/build_knowledge_graph.py --limit 5
```

Generate descriptions for a small sample:

```bash
python description-generation/main.py --limit 5
```

Run the judge over the latest generated description file:

```bash
python judge/judge.py
```

Expected generated output locations:

- `results/knowledge_graph/`
- `results/pr-description/<provider>/`
- `results/judge/<provider>/`

## Analysis Scripts

Human-study analysis can be run from the included inputs:

```bash
cd final-results
python3 scripts/analyze_human_evaluation.py
```

Description-level analyses require judged-output JSON files under
`final-results/data/description-data/`, as described in
`final-results/README.md`.
