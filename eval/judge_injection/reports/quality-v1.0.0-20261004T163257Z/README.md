# PromptShield quality run snapshot

This directory is a shareable results snapshot from the first PromptShield quality
experiment. It records dataset release **1.0.0** (`release-v1.0.0.jsonl`, SHA-256
`ffee101cd0e4de22b3386940cb0c1559267c2e7aa93c7f927a80c953a46e9b9d`), graded with
`ollama/llama3.2:3b` and PromptShield revision
`28b2670e65bbc29021ce739cbbd2449b03b1cdc8`.

Source run: `results/quality/20261004T163257Z`, created at
`2026-10-04T16:32:57Z` (UTC). The report was copied directly from that run so its
manually added attack examples are retained.

## Included artifacts

- `report.md` — results narrative, metrics, and attack examples
- `summary.json` — aggregate measurements
- `experiment_manifest.json` — run configuration, provenance, and integrity metadata
- `metrics.csv` — overall and grouped metrics
- `attack_types.csv` — per-family attack outcomes
- `confusion_matrices.csv` — filter confusion matrices
- `failures.csv` — missed attacks, false positives, and execution errors

This is a results snapshot, not a resumable run directory. Resume and report
regeneration require the original local run directory, which retains per-case results
and native Inspect logs. Those files, along with runtime state and caches, remain ignored
and are not included here.
