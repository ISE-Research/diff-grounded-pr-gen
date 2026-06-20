# Final Results Layout

This directory is organized into four areas:

- `data/human-data`
- `data/description-data`
- `scripts`
- `eval` (`eval/tables`, `eval/plots`)

## Data

### `data/description-data`
Six JSON files used for description evaluation:
- `descriptions-aidev-modes-raw-cmg-only-file-summaries-only-full-20260124-023308-461478-judge-20260125-010236-364060.json`
- `descriptions-parsed-modes-raw-cmg-only-file-summaries-only-full-20260124-010830-751653-judge-20260124-234250-571918.json`
- `results_aidev_c.json`
- `results_aidev_z.json`
- `results_parsed_c.json`
- `results_parsed_z.json`

Note:
- Files containing `parsed` correspond to the curated dataset from Trudeau et al. and are reported as `Trudeau` in evaluation outputs.

### `data/human-data`
Human study inputs:
- `Human Evaluation Survey- Pull Request Descriptions.csv`
- `survey_researcher_key.md`

## Scripts

### `scripts/analyze_descriptions.py`
Evaluates the six description JSON artifacts.

Outputs:
- `eval/tables/descriptions_summary.json`
- `eval/tables/descriptions_summary.csv`
- `eval/tables/descriptions_summary.md`
- `eval/plots/descriptions_summary.png`

### `scripts/analyze_human_evaluation.py`
Evaluates human rankings/preferences using the survey CSV and key.

Outputs:
- `eval/tables/human_evaluation_summary.json`
- `eval/tables/human_evaluation_summary.csv`
- `eval/tables/human_evaluation_summary.md`
- `eval/plots/human_evaluation_summary.png`

## Run

```bash
cd final-codebase/final-results
python3 scripts/analyze_descriptions.py
python3 scripts/analyze_human_evaluation.py
```
