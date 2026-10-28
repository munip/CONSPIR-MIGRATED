# <Experiment title>

<!-- Copy this to experiment_requests/<task>/<short-slug>/EXPERIMENT.md (or run
`stationhouse design new experiment_requests/<task>/<short-slug>/EXPERIMENT.md`) and fill in every
section before anything runs. Comments like this one are prompts: replace them. Check it with
`stationhouse design check <file>`. See "Designing an experiment" in CONTRIBUTING.md. -->

## Claim

<!-- One falsifiable sentence: "If X, then <metric> on <which items> moves by at least Δ compared with Y."
For example: "Fine-tuning Qwen3-4B on <data> lowers its divergence rate on the corpus by at least 5
points compared with the base model." -->

## Unit and metric

<!-- The unit: an item, a claim (all framings of one claim_id), a turn, a judge verdict. How many?
The metric (divergence rate, accuracy, κ against the reference labels, flip rate, ...), aggregated
how. Which judge grades it, with which rubric. What the metric does for a constant answer
(always hold, always diverge, always abstain). -->

## Controls

<!-- The trivial baselines: the base or unmodified model; a constant policy (always hold / always
diverge); the majority label. The matched control: an arm that differs from the treatment in exactly
one thing (same prompts, same length, same judge). If your method did nothing useful, what would it
still score? -->

## Data

<!-- Corpus commit (questions/corpus.jsonl), any fixture by path and SHA-256, the reference labels
(e.g. reference_v1.2). Every filter with the row count after it. How you keep the evaluation set
apart: split by claim_id, never by item, and say whether any training or prompt-writing data mentions
corpus claims. -->

## Noise

<!-- Samples per item (the baseline eval used k=5), seeds, temperature. How you'll estimate run-to-run
noise (rerun, bootstrap over claims). The smallest difference you'd treat as real. The pilot (a few
items, one seed) that checks the plumbing and doesn't count. -->

## Readout

<!-- Written before the run: the exact numbers you'll report, the pass rule, and the result that
would refute the claim. -->

## How to run

<!-- Models with exact ids/revisions, judge and rubric version, sampling settings, hardware, exact
commands, and what gets saved where. If you need compute, say what (and tag @stationhouse in the PR). -->

## Deviations

<!-- Filled in with the results: anything that changed from the plan above, and why. Leave the
sections above as they were. -->
