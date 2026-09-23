# Pending submissions

One file per pull request, named after your submission (e.g. `wtc7-collapse.jsonl`,
`south-asia-batch-1.jsonl`). One JSON object per line, one or more questions, every
`id` set to the literal `cb-pending`.

Nothing in this directory is part of the benchmark yet. After your PR is merged, a
maintainer runs `python scripts/assign_ids.py` on `main`, which assigns real ids,
appends the items to `questions/corpus.jsonl`, and deletes the pending file. See
[`CONTRIBUTING.md`](../../CONTRIBUTING.md).
