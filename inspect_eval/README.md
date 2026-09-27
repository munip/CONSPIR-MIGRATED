# ConspiracyBench on Inspect (pilot)

A first pass at running ConspiracyBench with [Inspect](https://inspect.aisi.org.uk/), built
alongside [`eval/`](../eval/) rather than replacing it, as agreed in #42. Nothing in `eval/`
changes; this folder can be deleted without affecting anything else.

It is deliberately a thin adapter. The judge prompt, the judge-output parsing and the rate
arithmetic are imported from the existing code, not reimplemented:

| Piece | Source |
|---|---|
| Judge prompt, per turn | `eval/judge_prompt.py` (`build_judge_prompt`) |
| Parsing the judge's reply | `eval/judge_reliability/run_baseline.py` (`strict_parse`, `permissive_parse`) |
| Divergence, abstention, concession, retreat | `eval/score_outcomes.py` |

So a result from here and a result from `eval/` differ only in the plumbing, which is what
a pilot should be able to show.

## Running it

```bash
pip install inspect-ai
inspect eval inspect_eval/conspiracybench.py --model <model-under-test> --model-role grader=<judge-model>
```

- The judge is a **required** model role. Without `--model-role grader=...` the run fails
  instead of quietly grading the model with itself.
- `-T include_pending=false` runs the merged corpus only. By default pending submissions in
  `questions/pending/` are included, with sample ids like `pending/wtc7-collapse:1`.
- `-T epochs=5` samples every item five times (see "Repeated sampling" in SCHEMA.md).
- `inspect view` opens the log viewer, which shows each item's full exchange, including every
  pressure turn and the judge's raw reply per turn.

## What it does

- **Dataset**: `questions/corpus.jsonl` plus `questions/pending/*.jsonl`. Each item becomes a
  sample with the question as input, the ground truth as target, and the whole item in
  metadata.
- **Solver** (`pressure_turns`): answers the question, then each of the item's
  `pressure_turns` in order, keeping the model's own replies in context. Items without
  pressure turns get a single generation.
- **Scorer** (`conspiracy_judge`): labels every answer `hold`, `diverge` or `abstain`. The
  opening answer is graded with the single-turn prompt, exactly as every existing item is;
  each later answer is graded with the preceding exchange as context. The judge runs at
  temperature 0. A reply the parser cannot read is recorded as `invalid` rather than
  guessed.
- **Metrics**: `divergence_rate` and `abstention_rate` over opening turns (the SCHEMA.md
  definitions), `concession_rate` and `retreat_rate` over multi-turn trajectories that open
  with `hold`, and `invalid_rate`. Trajectories containing `invalid` are excluded from every
  rate except `invalid_rate`.

## Why the score is a string like `hold>hold>diverge`

Each trajectory is stored as one scalar string, reduced across epochs with `collect`.
Checked against Inspect's source (0.3.269), the alternatives lose data once `epochs > 1`:

- `Score.metadata` is reduced to the first epoch's metadata.
- A dict-valued score is reduced key by key (`mode`, `mean` and so on), so turn 1 from one
  epoch can end up next to turn 2 from another, producing a trajectory no run ever produced.
- `collect` keeps every epoch's value, but only accepts scalar values.

Per-turn labels are also kept in `Score.metadata` for reading a single log, but the metrics
only use the value.

## Tests

```bash
python -m unittest discover -s inspect_eval -p "test_*.py"
```

Eight tests, no API calls: the model under test and the judge are both Inspect's `mockllm`,
scripted per test. They check that the judge sees byte-for-byte the prompts
`eval/judge_prompt.py` builds, that single-turn items get the single-turn prompt, that two
epochs with different outcomes both survive reduction, that unparseable judge replies are
reported rather than counted, and that a run without a judge fails.

## Not done yet

- No run against a real model or judge has been made.
- The repeated-sampling metrics from #38 (`worst_of_k`, `all_k_agree`, `pass_at_k`) are not
  wired in yet. The per-epoch trajectories they need are already kept, so this is a small
  follow-up using `summarize_samples` from `eval/score_outcomes.py`.
- Porting the judge-reliability runner (#50) is left for after this settles, as JoyeeChen
  suggested.
