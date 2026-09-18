# ConspiracyBench evaluation pipeline

Early, work-in-progress tooling for actually scoring a model against the corpus, as
opposed to the `scripts/` tools, which validate and maintain `questions/*.jsonl` itself.

- [`judge_prompt.py`](./judge_prompt.py) — builds the prompt an LLM judge sees for one
  `(item, model_answer)` pair, composing `ground_truth_answer` with the item's
  `diverges_if` / `does_not_diverge_if` / `notes` guidance when present. See
  [`SCHEMA.md`](../SCHEMA.md).
- [`score_outcomes.py`](./score_outcomes.py) — aggregates a list of judge outcomes
  (`hold` / `diverge` / `abstain`) into divergence rate (over answered items) and
  abstention rate (over all items). Divergence rate is `n/a` when every item is
  abstained, rather than a misleading 0%.

Nothing here calls a judge model yet — that's the next piece.
