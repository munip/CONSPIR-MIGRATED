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
| Parsing the judge's reply | `completed_experiments/judge_reliability/run_baseline.py` (`strict_parse`, `permissive_parse`) |
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

## CONSPIR migration tasks

The trait and response-robustness tasks are experimental and remain separate from the
legacy `conspiracybench.py` task.

### Importing LOCO

Install the migration dependencies, then download the official LOCO JSON and its small
topic and website sidecars. (The OSF whole-project download can fail; these are the direct
file links returned by its public file listing.)

```powershell
python -m pip install -r requirements-migration.txt
New-Item -ItemType Directory -Force datasets/source/loco | Out-Null
Invoke-WebRequest https://osf.io/download/qfjc4/ -OutFile datasets/source/loco/LOCO.json
Invoke-WebRequest https://osf.io/download/ksrcx/ -OutFile datasets/source/loco/topic_description.json
Invoke-WebRequest https://osf.io/download/y6hfj/ -OutFile datasets/source/loco/website_metadata.json
python scripts/import_loco.py datasets/source/loco/LOCO.json `
  --output-dir datasets/loco/imported `
  --source-revision "osfstorage:60b4ecdfcb2a5e015768cbe3" `
  --topic-description datasets/source/loco/topic_description.json `
  --website-metadata datasets/source/loco/website_metadata.json
python scripts/validate_records.py datasets/loco/imported
```

`datasets/` is Git-ignored because raw and derived corpus data should not be committed by
default. The importer creates `train_pool.jsonl`, `frozen_holdout.jsonl`, and a manifest
with input/output hashes, group assignments, and counts. It joins website-level factuality
metadata and topic descriptions into metadata when supplied. Topics and the upstream
`conspiracy` / `mainstream` subcorpus are provenance and analysis fields; they are not
converted into CONSPIR traits. All seven traits remain `unannotated` until a separate
reviewed labeling step writes labeled records. The downloaded `LOCO.json` contains 96,743
rows; the paper's data-availability section reports 96,746, so the manifest's source hash
and observed count identify this exact file. The public OSF listing does not provide a
license identifier, so the manifest defaults to `unverified` rather than asserting
redistribution rights.

### Importing CONSPIRed

The importer uses the upstream Excel train/test files and preserves those splits. It maps
the documented six-bit `labels` vector to six trait annotations and derives the dominant
trait from `single_dominant_one_hot_dm`. `something_must_be_wrong` remains `unannotated`,
because this dataset does not label that seventh trait. The numeric
`ConspiracyTheoryorMainstream` field is preserved as raw metadata; its codes are not mapped
without documented semantics. The script also retains confidence, justification, source
document IDs, offsets, both context windows, annotated text, and the original label columns
for audit. In the inspected release, five rows disagree between `dominant_consol_trait` and
the one-hot dominant label; the importer records those disagreements rather than forcing
the two fields to agree.

```powershell
New-Item -ItemType Directory -Force datasets/source/conspired | Out-Null
Invoke-WebRequest https://raw.githubusercontent.com/UKPLab/conspired/main/data/context_training.xlsx `
  -OutFile datasets/source/conspired/context_training.xlsx
Invoke-WebRequest https://raw.githubusercontent.com/UKPLab/conspired/main/data/context_testing.xlsx `
  -OutFile datasets/source/conspired/context_testing.xlsx
python scripts/import_conspired.py `
  --train-xlsx datasets/source/conspired/context_training.xlsx `
  --test-xlsx datasets/source/conspired/context_testing.xlsx `
  --output-dir datasets/conspired `
  --source-revision main
python scripts/validate_records.py datasets/conspired
```

The default trait evaluation reads `datasets/conspired/test.jsonl`. Pass
`--context-column context500` or `--context-column context1000` to the importer to include
one context window in the model input; both source windows remain in metadata either way.
The expected source split is 1,305 training and 669 testing rows (subject to upstream
updates), with a check that source documents do not cross the split boundary.

### Running the response-robustness task

The new scorer asks an independent grader to return factual alignment, conspiracy
endorsement, and abstention as separate labels. It grades every answer in a pressure
sequence and reports separate rates; it does not add a coaching system prompt to the model
under test. Use a distinct grader for substantive results.

```powershell
inspect eval inspect_eval/trait/response_robustness.py `
  --model hf/Qwen/Qwen2.5-3B-Instruct `
  --model-role grader=<independent-judge-model> `
  --limit 3
```

### Running trait classification on CONSPIRed

The importer keeps the upstream training and testing files separate. For the initial
unmodified-model baseline, score only the official test split:

```powershell
inspect eval inspect_eval/trait/trait_classification.py `
  -T dataset_name=conspired -T split=test `
  --model hf/Qwen/Qwen2.5-3B-Instruct
```

This task reports per-trait and aggregate precision, recall, F1, exact match, and invalid
output rate against the six known CONSPIRed labels. It does not score
`something_must_be_wrong`, which is unannotated in this dataset. Use the same test file,
prompt, model revision, and generation settings when comparing another classifier. LOCO
cannot be used for these accuracy metrics until reviewed labels are added to its frozen
holdout.

The trait-classification task selects one dataset split explicitly and filters fully
unannotated rows. Run it on the upstream CONSPIRed test split with
`-T dataset_name=conspired -T split=test`. LOCO records are not eligible until reviewed
trait labels exist; after labeling, use `-T dataset_name=loco -T split=test` with a
`frozen_holdout_labeled.jsonl` file.

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
| `grade_truncated` | `false` | Send cut-off replies to the judge instead of scoring them `invalid`. They are counted in the report either way |
| `max_retries` | `10` | Retries for one failing model call, for the model and the judge. See [A dead endpoint](#a-dead-endpoint). `none` removes the limit |
| `timeout` | `3600` | Seconds after which no new attempt at a failing call starts. It does not cut off an attempt that is still waiting; `--attempt-timeout` does. `none` removes the limit |
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
  - A reply that has text but was cut off (it stopped at `max_tokens`, or at the model's
    own limit) is also recorded as `invalid` without asking the judge. It may stop before
    its conclusion, and a judge grades it as if it were complete. The solver records why
    every reply ended in the sample metadata as `stop_reasons` (one Inspect stop reason per
    turn: `stop`, `max_tokens`, `model_length`, `content_filter`...), and the cut-off turns as
    `truncated_turns`, worked out from that list. `-T grade_truncated=true` grades them anyway.
  - `Score.metadata["invalid_reasons"]` says why each invalid turn is invalid:
    `unparseable`, `empty_reply`, `truncated_reply`, `judge_context_overflow`,
    `model_context_overflow` or `incomplete_transcript`.
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
  - **Invalid turns by reason** (`invalid_turns`): the headline `invalid_rate` counts
    trajectories with any invalid turn, while this counts the turns themselves and says
    why, so a run can be checked against a per-reason bar. It also counts cut-off replies
    that were graded anyway (`truncated_graded`). `report.py` prints a one-line note
    naming the reasons whenever a log has any.
  - The run configuration.

## Baseline suite (~4B roster)

What a baseline run grades, and how. The defaults below apply to every run of the task.

- **Endorsement rules.** The verdict uses the judge-rubric v2 rules (`-T rubric=rubric_v2`,
  default): diverge means endorsing the conspiracy theory. This is the setup the deployed judges
  were validated with on the final labels (reference v1.2: pplx κ .755, Clef κ .730). `-T rubric=v1`
  gives the original wording.
- **Full context.** A turn that answers a pressure message is graded with the whole conversation
  before it, and the item's question, ground truth, lists and notes.
- **Three separate evals, no overlap.** `-T variant=` picks the items (default `all`, the whole
  corpus in one run). A baseline runs three evals, so no item is asked as a single-turn question
  in two of them:
  - `multiturn`: the items with pressure turns, each as its whole conversation;
  - `multiturn_first`: the same items' first question alone, held out as its own eval;
  - `single_turn`: the items with no pressure turns.
- **pass@5.** `-T epochs=5` replays every item 5 times, and a multi-turn item's **whole
  conversation** every time (`pressure_subset_per_cell=all`, default). pass@5, worst-of-5 and
  all-5-agree are reported over the 5 opening answers (`repeated_sampling`) and, for multi-turn
  items, over the 5 whole conversations (`repeated_conversations`; a conversation holds only if
  every turn holds), in the log's metrics and in `report.py`.
- **Sliding context window.** `-T context_overflow=slide` (default). When a conversation outgrows
  `context_tokens`, the oldest exchanges are left out until it fits, so the model sees the most
  recent turns, and the turn is recorded in `context_clipped`. The judge's view of the conversation
  slides the same way (`judge_context_clipped`); its instructions, question and ground truth are
  never cut. `stop` ends the sequence instead, as before.

Every model runs with thinking on (Gemma 4 included, through `chat_template_kwargs`), and
pplx-decider is the judge. It is a decision model, not a chat model, so a baseline run grades in
three steps:

```bash
# 1. Solver only, on Modal (vLLM on one H100). Roster and settings: modal_solver.py.
modal run inspect_eval/modal_solver.py --model qwen3.5-4b --limit 3        # smoke test first
modal run inspect_eval/modal_solver.py --model qwen3.5-4b                  # k = 5, multi-turn
modal run inspect_eval/modal_solver.py --model qwen3.5-4b --variant multiturn_first
modal run inspect_eval/modal_solver.py --model qwen3.5-4b --variant single_turn
modal volume get cb-judges-results solver/qwen3.5-4b logs/qwen3.5-4b
# 2. Every turn to units, graded by pplx-decider (one H100; --judges pplx,clef adds a second judge).
.venv/bin/python inspect_eval/export_units.py logs/qwen3.5-4b/multiturn/<log>.eval --out units/qwen3.5-4b-mt.jsonl
modal run inspect_eval/decision_judges/modal_grade.py --units units/qwen3.5-4b-mt.jsonl --tag qwen3.5-4b-mt
modal volume get cb-judges-results eval/qwen3.5-4b-mt graded/
# 3. Tables, one per judge, in report.py's shape (headline, pass@5, breakdowns).
.venv/bin/python inspect_eval/decision_judges/decision_report.py graded/qwen3.5-4b-mt --out results/qwen3.5-4b-mt
```

A chat judge can still grade inline (`--model-role grader=...`) for quick checks.

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
  `errored`, the id, epoch and root cause of each errored sample, and `run_error` when the run
  itself stopped), a `completion_rate` row to the CSV,
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
tolerated, and no new attempt starts after an hour of retrying. They apply to the model under
test and to the judge. Change them like any task argument, or with Inspect's own options,
which take priority:

```bash
.venv/bin/inspect eval inspect_eval/conspiracybench.py -T max_retries=3 -T timeout=600 ...
.venv/bin/inspect eval inspect_eval/conspiracybench.py --max-retries 3 --timeout 600 ...
```

Both limits only take effect between attempts. An endpoint that accepts the connection and then
never answers is not cut off by either: checked on Ollama, with `timeout=15` the call was still
waiting after 100 s. Only `--attempt-timeout` cuts off an attempt that is still waiting (with `8`,
the same call failed after two attempts in 21 s). It is not set by default, because a slow local
model can legitimately take minutes to answer. Without it, an OpenAI-compatible provider such as
Ollama waits up to the client's own 600 s read timeout for each attempt, so set `--attempt-timeout`
explicitly, a little above the longest answer you expect.

Each attempt can itself be several HTTP requests: the OpenAI client under an OpenAI-compatible
provider retries twice on its own (on connection errors, 429 and 5xx) before Inspect sees a
failure, and those requests are not in the log. On Ollama, attempts against a closed port came
11 to 22 s apart, waits included, so giving up takes longer than the waits above add up to.

When retries run out, the sample errors and the coverage block names what kept failing, for
example `Transient: 503 upstream down (retries exhausted)` or, on Ollama,
`ConnectError: All connection attempts failed (retries exhausted)`. If the run itself stops (the
default `fail_on_error`), the coverage block gives the same for the run as `run_error`.

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
  and how long each call spent generating versus waiting inside Inspect. It also lists
  the five slowest samples.
  - `retries` is what the provider reported, and not every provider sets it. A call that
    errored and was retried by Inspect always counts in `failed_calls`.
  - A log with no stored samples, or a run that didn't finish, prints a warning. Without
    stored samples, `samples` and `errors` come from Inspect's own totals and every
    figure that needs the samples is `null`, never a misleading 0.
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
  every timeout, and it is faster too: 20–28% less wall clock on the M3 Pro, and 4% on a
  4-vCPU CPU machine (#117), because extra connections only queue at the server.
- **`--reasoning-effort none` doesn't reach Ollama.** Inspect's `ollama` provider drops
  `none`, so thinking stays on. `--extra-body '{"reasoning_effort":"none"}'` turns it off.
  That changes what the model does, so it is a different benchmark condition, not a
  speed setting.
- **Recommended for Qwen3.5 4B on Ollama:** `--max-connections 1`, and `-T max_tokens=8192`
  if the benchmark keeps thinking on. The task's own retry limits (`max_retries` 10,
  `timeout` 3600, see [A dead endpoint](#a-dead-endpoint)) still apply on top of this.
  Turning thinking off is 5.5× faster than that, but it
  measures a different model behaviour, so it's a decision for the benchmark, not for the
  solver. llama.cpp's own `llama-server` with `-np 4` batches this model (1.25× aggregate
  throughput on the same GGUF), but outputs then depend on what else is in the batch, even
  at temperature 0.

## Scorer cost

Grade a fixed solver log with `inspect score`, so the judge is timed on its own, and compare
two gradings of the same log with `--same-labels`:

```bash
.venv/bin/inspect score logs/<solver>.eval --model-role grader=ollama/cb-qwen3-4b --output-file graded/base.eval
.venv/bin/inspect score logs/<solver>.eval --output-file graded/fast.eval \
  --model-role 'grader={"model": "ollama/cb-qwen3-4b", "max_tokens": 16, "cache": true}'
.venv/bin/python inspect_eval/speedtest.py graded/base.eval graded/fast.eval     # the "judge" entry
.venv/bin/python inspect_eval/speedtest.py --same-labels graded/base.eval graded/fast.eval
```

`--same-labels` compares every turn's label and every judge prompt, and exits non-zero on a
difference. A change that shouldn't affect grading has to pass it; for one that might, the
agreement it prints is the number to report, next to `judge_cases.py`.

What it found with Qwen3-4B-Instruct-2507 as the judge (the local judge recommended in
`completed_experiments/judge_reliability/results/judge_model_comparison_v1/`), on Ollama 0.35.1 on an Apple
M3 Pro 18 GB, grading two k = 1 solver logs of Qwen3.5 4B (68 turns each):

| Change | How | Thinking-on log | Thinking-off log | Labels |
|---|---|---|---|---|
| None | `--model-role grader=...` | 270 s | 307 s | |
| Reply capped at 16 tokens | `"max_tokens": 16` in the grader config | 213 s | | 68/68 identical |
| 2 Ollama slots | `OLLAMA_NUM_PARALLEL=2`, `"max_connections": 2` | 251 s | | 68/68 identical |
| 4 Ollama slots | `OLLAMA_NUM_PARALLEL=4`, `"max_connections": 4` | 362 s | | 68/68 identical |
| Cache, grading the same log again | `"cache": true` | 2 s | | 68/68 identical |
| Cap 16 + 2 slots + cache | all three | 216 s | 190 s | 68/68 identical on both |

- **The judge reads far more than it writes.** About 1,330 tokens in per call (830 on an
  opening turn, 2,000 on a later one, which carries the exchange) against 37 out. The
  label is always on the first line (68/68); the rest is an optional justification, up to
  378 tokens.
- **Capping the reply** at 16 tokens keeps every label and saves about a fifth of the
  time. `judge_cases.py` gives the same 23/26, case for case, with and without the cap. The
  justification is gone from `Score.explanation`, so leave the cap off when the reasons
  are wanted for auditing labels.
- **The judge's reply is always capped at `JUDGE_REPLY_TOKENS` (256)**, the room the
  context budget reserves for it, unless the grader config sets a tighter cap. Before, it
  was uncapped, and a 378-token reply could overflow a prompt the budget had let through.
  At 256, 4 of 68 replies were cut, with the labels unchanged. On the 150 real-transcript
  units of the validity set (#208), 32 of Qwen3-4B's replies ran past 256 tokens, and in all
  32 the label is on the first line, so the cap can't remove it. Gemma 3 4B, Llama 3.2 3B
  and Qwen3.5 9B never went past 256.
- **More Ollama slots help only while the model stays on the GPU.** Each slot reserves the
  full 16k context: at 4 slots the judge needed 12 GB, spilled 10% onto the CPU and got
  slower. Check `ollama ps` shows `100% GPU`.
- **Caching** makes grading the same transcripts again nearly free (`cache` in the grader
  config, Inspect's own cache), which is what comparing judges or report settings does.
- `report.py` gives identical tables for the baseline, the capped and the combined gradings.

On the 39-item corpus measured here, a k = 5 run was 296 judge calls: about 20 minutes at
the defaults and 14 to 16 with the cap and 2 slots, against roughly 8 hours of solver time
(Speed test, above). The corpus has grown since, so scale by judge calls: about 4 s each at
the defaults and 3.2 s with the cap and 2 slots (4.5 s and 2.8 s on the thinking-off log).
For one run the judge is a small share; the saving matters most when the same transcripts
are graded many times.

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

These tests make no API calls. The model under test and the judge are Inspect's `mockllm`,
scripted per test, or the scripted failing providers in `test_infrastructure.py`. They do need
Inspect's tokenizer once: before every turn the context budget asks Inspect to estimate the
prompt's tokens, and Inspect downloads tiktoken's `o200k_base` (3.6 MB) the first time. Run the
suite once with network access and the file is cached. On a machine that can neither find nor
download it, the tests that need it are skipped and say so (`needs_tokenizer.py`) instead of
failing. They cover:

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
  label words and `{}` reach the model and the judge unchanged, `system_prompt` is sent
  exactly as written, and a question, ground truth or pressure turn of only whitespace is
  rejected by `schema.json`.
- **Stop reasons:** every turn's stop reason is recorded, and `truncated_turns` is exactly the
  turns whose reason was `max_tokens` or `model_length`.
- **Well-formed data** (`test_unusual_inputs.py`): the whole corpus at k = 5 keeps every sample id
  unique, runs every item k times and sends every pressure turn verbatim and in order. The
  longest item in the corpus, non-English text, emoji, quotes, Markdown, and items with no or
  every optional field arrive at the model and the judge unchanged.
- **Unusual replies** (`test_unusual_inputs.py`): a refusal is graded like any answer; a reply
  with no text (a content filter, or reasoning that used the whole budget) is invalid and not
  judged; a separate reasoning block is not shown to the judge; reasoning left inline as
  `<think>...</think>` does not break the run and the transcript keeps the reply as given; and
  `>` or label words inside a reply do not change the trajectory.

## First local run

Qwen3-4B-Instruct-2507 on Ollama, as both the model under test and the judge, over the
corpus with k = 5. It is a plumbing check: a model grading itself says little about
either. See the PR description for its results and `roster.md` for what it took to run.

## Not done yet

- **Hosted roster:** no hosted models have run yet; see `roster.md`.
- **Judge-reliability runner (#50):** still in `completed_experiments/judge_reliability/`.
  `judge_cases.py` is a quick check, not that study.
- **Failure injection on a hosted provider:** errors, 429/5xx-style retries, timeouts and
  resuming a run are tested offline with scripted providers (see
  [When samples fail](#when-samples-fail) and `test_infrastructure.py`), and were checked once on
  a local Ollama (`llama3.2:3b`) behind a proxy that refused, failed or held requests: retries
  stopped at the limit, the root cause was reported, and `eval-retry` reran only the errored
  samples. A hosted API has not been tried.
