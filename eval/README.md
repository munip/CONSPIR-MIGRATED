# ConspiracyBench evaluation pipeline

Early, work-in-progress tooling for actually scoring a model against the corpus, as
opposed to the `scripts/` tools, which validate and maintain `questions/*.jsonl` itself.

- [`judge_prompt.py`](./judge_prompt.py) — builds the prompt an LLM judge sees for one
  `(item, model_answer)` pair, composing `ground_truth_answer` with the item's
  `diverges_if` / `does_not_diverge_if` / `notes` guidance when present. See
  [`SCHEMA.md`](../SCHEMA.md). For an item with `pressure_turns`, pass the preceding
  exchange as `transcript=[{"role": ..., "content": ...}, ...]` so the judge grades the
  latest response with the pushback in view. Omitting it leaves the single-turn prompt
  byte-for-byte unchanged.
- [`score_outcomes.py`](./score_outcomes.py) — aggregates a list of judge outcomes
  (`hold` / `diverge` / `abstain`) into divergence rate (over answered items) and
  abstention rate (over all items). Divergence rate is `n/a` when every item is
  abstained, rather than a misleading 0%. `summarize_trajectories()` takes one list of
  per-turn outcomes per item and reports `concession_rate` (`hold` → `diverge`) and
  `retreat_rate` (`hold` → `abstain`) separately, plus `wavered` for items that diverge
  mid-conversation and recover by the last turn, which endpoint-only comparison hides.
- [`judge_cases.jsonl`](./judge_cases.jsonl) — a small regression/smoke set of
  `(item, model_answer, expected_label)` cases for exercising a judge implementation:
  clean holds, accurate extra context, false balance, direct denial, unsupported
  expansion, genuine uncertainty, evasive non-answers, and answer-embedded prompt
  injection attempting to override the expected label. Not a representative benchmark
  sample.

- [`test_judge_prompt.py`](./test_judge_prompt.py) and
  [`test_score_outcomes.py`](./test_score_outcomes.py) — `unittest` coverage for both,
  no dependencies beyond the standard library. Run them from the repo root with
  `python -m unittest discover -s eval -p "test_*.py"`. The single-turn prompt is pinned
  explicitly: a silent change there rescores every existing item, so `transcript=None`
  and `transcript=[]` are asserted to produce the untouched prompt.

Nothing here calls a judge model yet — that's the next piece.
