# CONSPIRED trait prompt and context experiment

## Claim

Across five document-grouped validation folds from the ConspirED training data,
Qwen 2.5 3B with both the `context1000` field and fixed labeled demonstrations
improves six-trait macro-F1 by at least 0.05 absolute compared with the same
six-trait prompt using only the annotated text and no demonstrations. A 2×2
factorial comparison will also estimate the separate context and demonstration
effects and their interaction.

## Unit and metric

Each annotated text snippet is one evaluation unit. Keep all snippets from the same
source `document_id` in one split and treat that document as the cluster for
uncertainty estimates.

The primary metric is macro-F1 across the six expert-labeled traits:
`contradictory`, `overriding_suspicion`, `nefarious_intent`, `persecuted_victim`,
`immune_to_evidence`, and `reinterpreting_randomness`. Compute F1 separately for
each trait, then give each trait equal weight in the average. Exclude
`something_must_be_wrong` from prompts and scoring because it is unannotated in
ConspirED.

Secondary readouts are per-trait precision, recall, F1, and support; micro-F1;
six-label exact match; and invalid-output rate. Also report candidate detection
precision, recall, and F1 using the derived rule that a snippet is a candidate
if at least one of the six scored traits is present. Mask unannotated labels in
all calculations. Keep invalid outputs in the evaluation: treat them as no
predicted traits for per-trait and candidate metrics, count their six-label
exact match as incorrect, and report their rate separately. Exact match is
calculated only over the six known labels.

## Controls

Use a 2×2 factorial design with four arms:

1. Annotated snippet only, no demonstrations.
2. Annotated snippet plus `context1000`, no demonstrations.
3. Annotated snippet plus fixed labeled demonstrations, no `context1000`.
4. Annotated snippet plus `context1000` and the same fixed demonstrations.

Across arms, hold the model and version, six-trait instructions, output format
and parser, decoding settings, validation items, and scoring procedure constant.
Only the presence of `context1000` and demonstrations changes. Within each
fold, use the same fixed demonstrations across all four arms. Select them only
from that fold's training-partition records, never from its validation fold or
the reserved test set. As non-model references, score constant predictions of
all six traits present and all six absent on the same out-of-fold items. The
all-present rule has macro-F1 about 0.331 on the full training labels; use its
out-of-fold score for comparisons, not this training-set value.

## Data

Use the official ConspirED training artifact at
`datasets/conspired/train.jsonl` (1,305 source records; SHA-256
`149f80d46018fef0c7ffc0b3be0d9cc6ee33d146b9a7edb9d87d6eb3a7310f45`), containing
31 unique `source.document_id` groups. Assign groups to five deterministic
validation folds with seed 42, greedily balancing snippet counts while keeping
each document intact; every eligible row is evaluated once. The importer stores `context1000` under
`metadata.context1000`. One source record
(`conspired-train-0d5c97922cbc7dd17b82`) has no nonblank context; exclude it from
all four arms so every paired item receives the context treatment where assigned.
This leaves 1,304 evaluation rows across the same 31 document groups. Record the
exact fold and demonstration manifest before generating any model outputs.

Eligible rows must have nonblank `text`, a `source.document_id`, nonblank
`metadata.context1000`, and known `present`/`absent` annotations for all six
scored traits. All 1,305 source records have unique IDs and complete six-trait
labels; the one row without context is excluded from every arm. For each fold,
select one distinct training-pool example
with each scored trait present, plus one all-negative example if available.
Prefer distinct source documents when possible. Demonstrations use the
annotated snippet only, not `context1000`, and are held constant across that
fold's four arms. Do not use the reviewed 669-record test set for prompt or
example selection.

## Noise

Use deterministic decoding with `temperature=0` and one generation per
validation item in each arm. Estimate uncertainty in arm differences with a
paired 95% bootstrap interval that resamples source documents, keeping all
snippets from a sampled document together. This captures uncertainty from the
validation documents without adding repeated stochastic generations.

## Readout

The primary comparison is arm 4 (snippet + `context1000` + demonstrations)
against arm 1 (snippet only, no demonstrations). Support the claim only if the
observed macro-F1 gain is at least 0.05 and the lower bound of its paired 95%
document-bootstrap interval is also at least 0.05. In addition, the combined
arm must beat the all-six-present baseline in macro-F1, with the lower bound of
the paired 95% interval for that difference above zero. If the combined arm
beats arm 1 but not the constant baseline, report the relative gain but conclude
that the classifier objective was not met. The 0.05-improvement claim is
refuted if the upper bound of its paired 95% interval is below 0.05; if the
interval includes 0.05 but its lower bound does not reach that threshold, call
the result inconclusive rather than treating it as success or refutation.

Report macro-F1 and the interval for all four arms, the separate context and
demonstration effects and their interaction, per-trait precision/recall/F1 and
support, micro-F1, six-label exact match, derived candidate-detection
precision/recall/F1, constant-baseline scores, and invalid-output rate. Invalid
outputs remain in the denominators as specified above. State the model and
decoding settings, split manifest, and any missing/undefined metrics. A low
score does not invalidate the experiment; it indicates model performance on
this task. Do not use the reserved test set to tune prompts or change the
decision rule.

## How to run

Model: `hf/Qwen/Qwen2.5-3B-Instruct`; record and pin the resolved model
revision before running. Use deterministic greedy decoding (`do_sample=False`);
the manifest records the requested `temperature=0`, while the Hugging Face
provider omits the temperature argument because Transformers rejects zero.
Load with `--dtype auto`, which uses the checkpoint's declared BF16 dtype for
the pinned revision; this is not quantization. Record the inference hardware,
dtype, quantization, and concurrency settings.

The separate `run_trait_prompt_experiment.py` runner implements the four arms
and five grouped folds. Before the full experiment, a one-record smoke run can
check provider access, prompt construction, scoring, and artifact writing. It
runs one positive validation example from fold 1 through all four arms (four
model generations) and is not a performance estimate. Run it with separate,
fresh output directories:

```powershell
python scripts/run_trait_prompt_experiment.py --model hf/Qwen/Qwen2.5-3B-Instruct --model-revision <pinned-revision> --run-environment "<hardware; dtype=auto (BF16); quantization=none; concurrency>" --dtype auto --folds 5 --seed 42 --temperature 0 --smoke --results-dir results/trait-prompt-context-smoke --log-dir logs/trait-prompt-context-smoke
```

For a bounded runtime pilot, `--pilot-samples N` selects N deterministic
validation examples from fold 1 and evaluates the same examples in all four
arms, for `4 × N` generations. For example, 25 examples require 100
generations. This pilot is useful for estimating throughput and checking the
reporting path, but its subset is not a substitute for the grouped five-fold
experiment and must not be used to support or reject the experiment claim.
Inspect's `--max-samples` controls sample concurrency (not the total sample
count); the runner exposes it as `--max-samples` and defaults to 1:

```powershell
python scripts/run_trait_prompt_experiment.py --model hf/Qwen/Qwen2.5-3B-Instruct --model-revision <pinned-revision> --run-environment "<hardware; dtype=auto (BF16); quantization=none; concurrency>" --dtype auto --folds 5 --seed 42 --temperature 0 --pilot-samples 25 --max-samples 5 --results-dir results/trait-prompt-context-pilot --log-dir logs/trait-prompt-context-pilot
```

The full experiment, which evaluates all 1,304 eligible rows in each arm, is a
separate run:

```powershell
python scripts/run_trait_prompt_experiment.py --dataset datasets/conspired/train.jsonl --model hf/Qwen/Qwen2.5-3B-Instruct --model-revision <pinned-revision> --run-environment "<hardware; dtype=auto (BF16); quantization=none; concurrency>" --dtype auto --folds 5 --seed 42 --temperature 0 --results-dir results/trait-prompt-context --log-dir logs/trait-prompt-context
```

Save each run's fold manifest and report under its results directory, and
preserve Inspect logs under its log directory. Smoke and pilot metrics and
bootstrap criteria must not be used to support or reject the experiment claim.
The reserved test set remains off limits for tuning.

## Deviations

<!-- Record any changes after the run. Do not revise the agreed design stages retroactively. -->
