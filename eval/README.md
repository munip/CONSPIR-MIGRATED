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
  `summarize_samples()` takes k independent opening-turn outcomes per item (`DEFAULT_K =
  5`) and reports `worst_of_k`, `all_k_agree` and `pass_at_k`, each as a count, rate and
  95% Wilson interval, plus the pooled per-sample rates. See
  [Repeated sampling](../SCHEMA.md#repeated-sampling-k--5) for why pass@k is not the
  headline.
- [`judge_cases.jsonl`](./judge_cases.jsonl) — a small regression/smoke set of
  `(item, model_answer, expected_label)` cases for exercising a judge implementation:
  clean holds, accurate extra context, false balance, direct denial, unsupported
  expansion, genuine uncertainty, evasive non-answers, and answer-embedded prompt
  injection attempting to override the expected label. Not a representative benchmark
  sample.

- [`judge_cases_multiturn.jsonl`](./judge_cases_multiturn.jsonl) — multi-turn companion
  to `judge_cases.jsonl` for items with `pressure_turns`. Each case adds a `transcript`
  (the question, earlier model answers, and the item's own pressure turns in order) and
  grades only the final `model_answer`. Covers concession after earlier holds, partial
  retreat into false balance, an accurate caveat that should still hold, disengagement
  that should abstain, recovery after an earlier concession, and rebutting a conspiracy
  claim by name. Kept as a separate file because the judge-reliability runners expect
  exactly the 15 single-turn cases in `judge_cases.jsonl`.
  [`test_judge_cases_multiturn.py`](./test_judge_cases_multiturn.py) checks every case
  is well formed without calling a judge.
- [`test_judge_prompt.py`](./test_judge_prompt.py) and
  [`test_score_outcomes.py`](./test_score_outcomes.py) — `unittest` coverage for both,
  no dependencies beyond the standard library. Run them from the repo root with
  `python -m unittest discover -s eval -p "test_*.py"`. The single-turn prompt is pinned
  explicitly: a silent change there rescores every existing item, so `transcript=None`
  and `transcript=[]` are asserted to produce the untouched prompt.

Nothing here calls a judge model yet — that's the next piece.
