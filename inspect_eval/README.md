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
| `cb_dataset.py` | Corpus items (and, if asked, pending ones) as samples; the stratified pressure subset |
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
- **Only the merged corpus is evaluated by default.** Pending items aren't part of the
  benchmark yet; `-T include_pending=true` adds them, with sample ids like
  `pending/wtc7-collapse:1`. Every item is validated against `schema.json` on load, so a
  broken pending draft stops the run with an error naming its file and line.
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
| `context_tokens` | `16384` | Token budget every prompt must fit in. See [Context budget](#context-budget). `none` turns it off |
| `judge_context_tokens` | same as `context_tokens` | Budget for the judge, when its context differs from the model under test's |
| `max_retries` | `10` | Retries for one failing model call, for the model and the judge. See [A dead endpoint](#a-dead-endpoint). `none` removes the limit |
| `timeout` | `3600` | Seconds spent retrying one call before giving up. `none` removes the limit |
| `system_prompt` | none | Sent as a system message when set |
| `include_pending` | `false` | Also include `questions/pending/` |

Rate limits and concurrency are Inspect's own options (`--max-connections`, `--fail-on-error`).
`--max-retries` and `--timeout` override the task's `max_retries` and `timeout`.

## What it does

- **Dataset**: `questions/corpus.jsonl` (plus `questions/pending/*.jsonl` with
  `include_pending=true`). Each item becomes a
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
  - An empty answer from the model under test is recorded as `invalid` without asking the
    judge. A thinking model can spend all of `max_tokens` reasoning and return no text,
    and a judge given an empty response still returns a label: Gemma 3 4B said `diverge`
    for 32 of 34 such Qwen3.5 4B replies.
  - `Score.metadata["invalid_reasons"]` says why each invalid turn is invalid:
    `unparseable`, `empty_reply`, `judge_context_overflow` or `model_context_overflow`.
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

## Context budget

Every prompt has to fit in `context_tokens`, checked before it is sent
([`cb_budget.py`](./cb_budget.py)). A prompt that would not fit is **not sent**, and its
turn is scored `invalid` with a reason:

- **Model under test:** if the conversation so far plus `max_tokens` of reply would not
  fit, the pressure sequence stops at that turn (`model_context_overflow`).
- **Judge:** if the judge prompt plus room for its reply would not fit, the judge is not
  called for that turn (`judge_context_overflow`).

This exists because an over-long prompt does not always fail. Ollama can cut it
silently, and it drops the *start*. For the judge, the start is the instructions, the
question, the ground truth and the `diverges_if` checklist, so it would return a
confident label chosen without the ground truth. Ollama also picks its default context
from available memory, so the same run can error on one machine and truncate silently
on another.

The budget is a value the task sets, not something read from the provider, and it must
not exceed the context the models are loaded with (`num_ctx` in the Modelfile, 16384).
It is checked from Inspect's estimate (`Model.count_tokens`), which on a late-turn judge
prompt was 866 against 829 tokens actually counted by Ollama (qwen3:1.7b). Checking
afterwards from the provider's reported input tokens does not work: Ollama reports only
tokens that were not already in its prompt cache, so later turns report about one token.

## When samples fail

A run is hours of model calls, so some will fail. Inspect decides what happens, and the
choices differ in how visible a failure is:

| Option | What happens to a sample that errors |
|---|---|
| default (`fail_on_error=True`) | The run stops on the first error. Loud, and costly for a long run. |
| `--retry-on-error N` | The sample is re-run from the start, up to N times. The transcript restarts cleanly, with no turn repeated or skipped. **Recommended for long runs.** |
| `--fail-on-error 0.05`, `--no-fail-on-error`, `--continue-on-fail` | The run continues. Inspect still reports `success`, and the errored sample has no score. |
| `--score-on-error` | The cut-off transcript is graded as well. |

With the last two, rates are computed over the samples that completed, so the results do not
say by themselves that anything is missing. Two things make that visible:

- **Coverage.** `report.py` adds a `coverage` block to the JSON (`planned`, `completed`,
  `errored` and the id, epoch and root cause of each errored sample), a `completion_rate` row to the CSV,
  and prints a warning to stderr. The exit status is unchanged, since tolerating failed samples
  is the point of those options.
- **`incomplete_transcript`.** The solver records the turns it meant to play in the sample
  metadata (`turns_planned`). If the scorer grades fewer turns than that, the missing turn is
  labelled `invalid` with the reason `incomplete_transcript`. The sample then stays out of the
  rates, like any other invalid trajectory, instead of reading as a shorter conversation that
  held. A sample that was meant to be short is not flagged: a single-turn item, a later epoch
  that plays only the opening turn, or a conversation stopped for lack of context (that one keeps
  `model_context_overflow`).

### A dead endpoint

Inspect retries a failing model call for ever unless told otherwise, and the waits grow (3 s,
doubling, up to 30 minutes), so a dead endpoint stalls a run without ever failing it.
`--retry-on-error` and `--fail-on-error` cannot help, because the call never returns.

The task therefore sets `max_retries=10` and `timeout=3600`: about 50 minutes of outage is
tolerated, and a call is given up on after about an hour of retrying. They apply to the model under
test and to the judge. Change them like any task argument, or with Inspect's own options,
which take priority:

```bash
.venv/bin/inspect eval inspect_eval/conspiracybench.py -T max_retries=3 -T timeout=600 ...
.venv/bin/inspect eval inspect_eval/conspiracybench.py --max-retries 3 --timeout 600 ...
```

A single attempt that hangs is cut off by `--attempt-timeout`, which is not set by default: a
slow local model can legitimately take minutes to answer. When retries run out, the sample errors
and the coverage block names what kept failing, for example
`Transient: 503 upstream down (retries exhausted)`.

## Speed test

[`speedtest.py`](./speedtest.py) reads Inspect logs and reports where a run's time went. It
makes no model calls, so any log can be rechecked later.

```bash
# 1. Baseline: the solver alone, k = 1, every setting recorded in the log.
.venv/bin/inspect eval inspect_eval/conspiracybench.py --no-score \
  --model ollama/cb-qwen3.5-4b --model-role grader=<judge> --log-dir logs/speed-baseline
# 2. Change one thing (here, concurrency) and run again into a new folder.
.venv/bin/inspect eval inspect_eval/conspiracybench.py --no-score --max-connections 1 \
  --model ollama/cb-qwen3.5-4b --model-role grader=<judge> --log-dir logs/speed-mc1
# 3. Compare, and check the model was sent the same thing.
.venv/bin/python inspect_eval/speedtest.py logs/speed-baseline/*.eval logs/speed-mc1/*.eval
.venv/bin/python inspect_eval/speedtest.py --same-prompts logs/speed-baseline/*.eval logs/speed-mc1/*.eval
```

- **`--no-score`** times the solver without the judge. Score the same log afterwards
  with `inspect score` to time the judge on its own.
- **Per log**, it prints the wall clock and run throughput, and the model-under-test calls
  split by single- vs multi-turn item and opening vs pressure turn: calls, failed calls,
  retries, tokens in and out, turns that hit `max_tokens`, **empty answers**, latency,
  and how long each call spent generating versus waiting inside Inspect.
- **`--same-prompts`** compares the HTTP request body each turn sent to the provider, byte
  for byte, after checking that every earlier reply in it is the reply the model actually
  gave in that run. It exits non-zero on any difference. A speed change must pass it.
  Inspect resends a reply with one leading newline added, the same in every run; the
  check expects it.
- Record the hardware, the Ollama version and its settings next to the numbers. Numbers
  from different machines aren't comparable.

What it found on Qwen3.5 4B with Ollama 0.35.1 (Apple M3 Pro 18 GB, k = 1, 39 items, 68
turns):

- **Thinking uses up `max_tokens`, and that changes the results.** Qwen3.5 4B thinks by
  default. At `max_tokens` 2048, 53 of 68 turns hit the limit: 34 replies were empty and
  19 were cut off. When this was measured the judge was still asked to grade an empty
  reply, and Gemma 3 4B labelled 32 of 34 of them `diverge`. Opening-turn divergence was
  49% (19/39) at 2048, against 18% (7/39) at 8192 and 21% (8/39) with thinking off. The
  scorer now labels an empty reply `invalid` (`empty_reply`) without asking the judge,
  so it no longer counts as `diverge`, but it still drops out of the rates. Check
  `empty_answers` before trusting a run on a thinking model.

  | Thinking, `max_tokens` (k = 1, `--max-connections 1`) | Wall clock | Empty / cut off | Opening divergence | k = 5 projected |
  |---|---|---|---|---|
  | on, 2048 | 72 min | 34 / 19 | 49% | 5.2 h |
  | on, 8192 | 108 min* | 0 / 0 | 18% | 7.8 h |
  | off (`--extra-body`, below) | 19.5 min | 0 / 0 | 21% | 1.3 h |

  \*117 min including a 9-minute Ollama restart; Inspect retried and no turn was lost. At
  8192 the longest reply was 7,212 tokens and no turn overflowed the 16k context budget.
  The judge was Gemma 3 4B (`num_ctx` 16384) via `inspect score`, which took 1.5 to 3.5
  minutes per log. k = 5 is 195 opening and 101 pressure turns, timed at each run's mean.
- **Ollama serves Qwen3.5 one request at a time.** It ignores `OLLAMA_NUM_PARALLEL` for
  this architecture (`model architecture does not currently support parallel requests`).
  With Inspect's default of 10 connections, the queue outlasts the OpenAI client's 600 s
  timeout: 51 requests failed and were retried. Use **`--max-connections 1`**. It removes
  every timeout at the same throughput.
- **`--reasoning-effort none` doesn't reach Ollama.** Inspect's `ollama` provider drops
  `none`, so thinking stays on. `--extra-body '{"reasoning_effort":"none"}'` turns it off.
  That changes what the model does, so it is a different benchmark condition, not a
  speed setting.
- **Recommended for Qwen3.5 4B on Ollama:** `--max-connections 1`, and `-T max_tokens=8192`
  if the benchmark keeps thinking on. Turning thinking off is 5.5× faster than that, but it
  measures a different model behaviour, so it's a decision for the benchmark, not for the
  solver. llama.cpp's own `llama-server` with `-np 4` batches this model (1.25× aggregate
  throughput on the same GGUF), but outputs then depend on what else is in the batch, even
  at temperature 0.

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
- **Awkward text:** a valid question holding U+2028 / U+2029 / U+0085 loads as one line, a lone
  surrogate is rejected at load time, chat-template tokens, fake `User:` / `Model:` lines,
  label words and `{}` reach the model and the judge unchanged, and `system_prompt` is sent
  exactly as written.

## First local run

Qwen3-4B-Instruct-2507 on Ollama, as both the model under test and the judge, over the
corpus with k = 5. It is a plumbing check: a model grading itself says little about
either. See the PR description for its results and `roster.md` for what it took to run.

## Not done yet

- **Hosted roster:** no hosted models have run yet; see `roster.md`.
- **Judge-reliability runner (#50):** still in `eval/judge_reliability/`.
  `judge_cases.py` is a quick check, not that study.
- **Failure injection on a real provider:** errors, 429/5xx-style retries, timeouts and
  resuming a run are tested offline with scripted providers (see
  [When samples fail](#when-samples-fail) and `test_infrastructure.py`); a real provider has
  not been tried.
