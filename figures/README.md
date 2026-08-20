# Canonical Paper Figures

This directory contains the camera-ready figure PDFs used by the paper. The
top-level files are the canonical versions for the replication package.

| File | Role in the paper/artifact |
| --- | --- |
| `architecture.pdf` | End-to-end pipeline architecture: cached PRContext construction, optional CMG, optional file-level summaries, and final PR-description generation. |
| `architecture.png` | README preview derived from `architecture.pdf`; use the PDF as the canonical paper figure. |
| `cached-artifact-schema.pdf` | Schema-level view of reconstructed PRContext records. |
| `cmg.pdf` | Commit-message generation and quality-gating component. |
| `file-diff-summarization.pdf` | File-diff selection, cleaning, and summarization component. |
| `motivating-example.pdf` | Motivating comparison among the original PR description, raw zero-shot baseline, and full diff-grounded approach. |

The `old-1/` and `old-2/` subdirectories contain earlier figure versions for
internal provenance only. They are not the canonical paper figures.
