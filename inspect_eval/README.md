# ConspiracyBench on Inspect

Runs ConspiracyBench with [Inspect](https://inspect.aisi.org.uk/), built alongside
[`eval/`](../eval/) rather than replacing it, as agreed in #42. It started as the pilot in
#54 and now covers the solver and scorer infrastructure: repeated sampling per SCHEMA.md,
metadata for breakdowns, the repeated-sampling metrics, and a results table with intervals.
Nothing in `eval/` changes.

It is still a thin adapter. The judge prompt, the judge-output parsing and the rate
arithmetic are imported from the existing code, not reimplemented:

| Piece | Source |
|---|---|
| Judge prompt, per turn | `eval/judge_prompt.py` (`build_judge_prompt`) |
| Parsing the judge's reply | `eval/judge_reliability/run_baseline.py` (`strict_parse`, `permissive_parse`) |
| Divergence, abstention, concession, retreat, worst-of-k, all-k-agree, pass@k, Wilson intervals | `eval/score_outcomes.py` |

| File | What it holds |
|---|---|
| `conspiracybench.py` | The task entry point and its run configuration |
| `cb_dataset.py` | Corpus and pending items as samples; the stratified pressure subset |
| `solver/` | `pressure_turns`: the opening question, then the pressure sequence |
| `scorer/` | `conspiracy_judge`: one hold / diverge / abstain label per turn |
| `cb_metrics.py` | The rates Inspect prints and stores in the log |
| `report.py` | A per-model results table from a log, with intervals and breakdowns |
| `judge_cases.py` | A check of a judge model against `eval/judge_cases*.jsonl` |
| `roster.md` | Model strings, provider setup and notes from the local run |

## Running it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python inspect-ai openai
.venv/bin/inspect eval inspect_eval/conspiracybench.py -T epochs=5 \
  --model <model-under-test> --model-role grader=<judge-model>
.venv/bin/python inspect_eval/report.py logs/<log>.eval --out results/
```

- **The judge is a required model role.** Without `--model-role grader=...` the run fails
  instead of quietly grading the model with itself. Check the judge first with
  `inspect eval inspect_eval/judge_cases.py --model <judge-model>`.
- **`-T epochs=5`** is the benchmark run (SCHEMA.md, k = 5). The default of one epoch is
  for quick checks.
- **Pending items are included by default.** `-T include_pending=false` runs the merged
  corpus only; pending sample ids look like `pending/wtc7-collapse:1`.
- **`inspect view`** opens the log viewer, with every turn, the judge's raw reply per turn,
  and the item metadata to filter on.
- For a local run on Ollama, see [`roster.md`](./roster.md). The context length needs raising
  from Ollama's 4096 default.

### Run configuration

These task arguments are recorded in the log, as are the model, the judge, and the git commit:

| Argument | Default | Meaning |
|---|---|---|
| `epochs` | `1` | Samples of every opening turn (k) |
| `pressure_subset_per_cell` | `1` | Multi-turn items per category × difficulty band whose pressure sequence repeats every epoch |
| `subset_seed` | `0` | Seed for choosing that subset; the chosen ids are in the log as the solver's `pressure_ids` |
| `temperature` | model default | SCHEMA.md samples "at the model's normal sampling temperature" |
| `max_tokens` | `2048` | Per model turn. 1024 cut off Qwen3-4B's multi-turn answers |
| `system_prompt` | none | Sent as a system message when set |
| `include_pending` | `true` | Include `questions/pending/` |

Retries, rate limits and concurrency are Inspect's own options (`--max-connections`,
`--max-retries`, `--timeout`, `--fail-on-error`).

## What it does

- **Dataset**: `questions/corpus.jsonl` plus `questions/pending/*.jsonl`. Each item becomes a
  sample with the question as input and the ground truth as target. The whole item is kept
  in metadata, along with `category`, `band` (from `difficulty.band`), `framing`,
  `claim_id`, `region`, `language` and `multiturn`, for filtering and breakdowns.
- **Solver** (`pressure_turns`): answers the question, then each of the item's
  `pressure_turns` in order, keeping the model's own replies in context. This follows
  SCHEMA.md's repeated sampling. Every epoch samples the opening turn. The pressure
  sequence plays in epoch 1 for every multi-turn item, and in every epoch only for a fixed
  subset stratified by category and difficulty band.
- **Scorer** (`conspiracy_judge`): labels every answer `hold`, `diverge` or `abstain`. The
  opening answer is graded with the single-turn prompt, exactly as every existing item is.
  Each later answer is graded with the preceding exchange as context. The judge runs at
  temperature 0.
  - Parsing tries three things in order: the whole reply as a label, then the first line
    as a label, then `permissive_parse` near the end. Small judges often give the label
    first and then a justification.
  - Leftover `<think>` reasoning is removed before parsing.
  - A reply that can't be read is recorded as `invalid`, never guessed.
- **Metrics in the log**: these rates are computed over the opening turns of every epoch
  (k × items):
  - `divergence_rate` and `abstention_rate`.
  - `concession_rate` and `retreat_rate`, over every multi-turn sequence in the log.
  - `invalid_rate`.
  - `worst_of_k`, `all_k_agree` and `pass_at_k`, over items with k valid opening samples
    (`nan` for one epoch).
  - Trajectories containing `invalid` are excluded from every rate except `invalid_rate`.
- **`report.py`**: reads each sample's epoch from the log. Inspect's metrics can't, because
  `collect` doesn't keep epoch order. For each model it writes `<model>.json` and
  `<model>.csv` containing:
  - Headline rates with 95% Wilson intervals.
  - Concession and retreat over epoch-1 sequences only (one per item). The subset's
    repeats are reported apart as `pressure_repeats`.
  - worst-of-k, all-k-agree and pass@k.
  - **Clustered intervals** (`ci95_clustered`): a Wilson interval on an effective sample
    size estimated by resampling whole claims, so variants of one `claim_id` and the k
    epochs of one item aren't counted as independent. At a 0% or 100% rate it counts one
    observation per claim.
  - Breakdowns by category, band, framing, region and single- vs multi-turn.
  - The run configuration.

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
.venv/bin/python -m unittest discover -s inspect_eval -p "test_*.py"
```

These 24 tests make no API calls. The model under test and the judge are both Inspect's
`mockllm`, scripted per test. They cover:

- **Judge prompts:** the judge sees byte-for-byte the prompts `eval/judge_prompt.py` builds,
  and single-turn items get the single-turn prompt.
- **Pressure turns:** they arrive verbatim and in order. Later epochs send only the opening
  question, and subset items repeat the whole sequence.
- **Epochs:** all epochs survive reduction.
- **Repeated-sampling metrics:** worst-of-k, all-k-agree and pass@k match their definitions.
- **Metadata and subset:** metadata reaches the log, and the stratified subset is
  deterministic.
- **Parsing:** reasoning and leading labels are parsed correctly. Unreadable replies are
  reported, not counted.
- **Reports:** `report.py` produces the table from a real log, with intervals that widen
  for shared claims.
- **Judge role:** a run without a judge fails.

## First local run

Qwen3-4B-Instruct-2507 on Ollama, as both the model under test and the judge, over the
corpus with k = 5. It is a plumbing check: a model grading itself says little about
either. See the PR description for its results and `roster.md` for what it took to run.

## Not done yet

- **Hosted roster:** no hosted models have run yet; see `roster.md`.
- **Judge-reliability runner (#50):** still in `eval/judge_reliability/`.
  `judge_cases.py` is a quick check, not that study.
- **Failure injection:** there's no test yet for API errors or timeouts on a real provider
  (the "Run experiments with solver infrastructure" task).
