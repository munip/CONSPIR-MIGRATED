# Legacy to v2 question-answer migration parity

## Claim

For the same Station House QA corpus and the same captured model responses,
converting each legacy record to v2 `question_answer` preserves every
evaluation-relevant field and yields identical per-item and aggregate
hold/diverge/abstain scores.

## Unit and metric

One paired unit is each Station House QA record in its legacy representation
and its v2 `question_answer` representation. Compare every evaluation-relevant
field and the Inspect input/target produced from each. Reuse identical captured
model responses and judge outputs, then require exact per-item
`hold`/`diverge`/`abstain` agreement and identical aggregate counts and rates.
Report record counts, conversion/schema-validation failures, field mismatches,
and score mismatches. Because this is a deterministic paired compatibility
check, do not use a confidence interval.

## Controls

Process each source record through the existing legacy loader and through the
v2 `question_answer` validator/loader. The v2 conversion may add
`schema_version` and `task_type`, but must not alter question, answer,
provenance, item, or pressure-turn content. Hold the model prompt, model
response, judge response, rubric, and scorer version fixed by reusing the same
captured outputs. Do not rerun or regrade between representations.

The legacy loader is the migration baseline; report its `hold`/`diverge`/
`abstain` counts and rates as the reference for v2. For score context, a
constant always-abstain policy yields 100% `abstain` and 0% `hold`/`diverge`;
this trivial baseline does not change the parity pass rule.

## Data

Use all 315 unique records in `questions/corpus.jsonl` (SHA-256
`22c21ee342796bb1a007be943ddff60ed7d9ece68ed06f381e72411e97fe027b`); do not
include pending submissions. Apply no row filters. Convert each record by
adding only `schema_version: "2.0"` and `task_type: "question_answer"`, preserving
all existing fields including `consensus_as_of`.

Pin the legacy `schema.json` at SHA-256
`3e3a074f70d6c15e53fe9f879055960b860fdefec6aca4aaa81dd0693b57fc70` and the
corrected `schemas/conspir/v2_schema.json` at SHA-256
`2757ade31e8789535dca11b15c79a5c5311222e311c5ac0ca6dca085db9c9e81`. Capture
one baseline QA evaluation's model responses and judge outputs once, then replay
that exact cache through both schema paths. Pin the model, judge, rubric, and
evaluation settings before that run.

No train/validation/test split is used: the full 315-record corpus is the
paired compatibility set, no records are copied to another set, and outputs
are not used for prompt tuning.

## Noise

Use one baseline QA run and cache each model response and judge label. Replay
the identical cache through both schema paths. Do not repeat model or judge
calls; stochasticity in the baseline is shared by the paired paths, and no
confidence interval is needed for exact parity.

## Readout

Parity passes only if all 315 records validate, no legacy field is lost or
changed, each legacy/v2 pair produces identical Inspect input and target, and
the same cached responses and judgments yield identical per-item labels and
aggregate `hold`/`diverge`/`abstain` counts and rates. Any conversion or
validation failure, content loss, or score mismatch refutes parity. Report
mismatch counts and examples grouped by field or processing stage.

## How to run

The `cb_dataset.conspiracybench_v2_dataset` loader validates the v2
`question_answer` representation after lossless conversion. The parity runner
reads an existing Inspect QA eval log, exports its model messages and cached
judge outputs, and maps the same cache through both schema representations.
It makes no model or judge calls. First run one baseline QA evaluation with a
pinned model, grader role, rubric, and settings; the parity runner verifies the
model, grader, and rubric supplied on its command line against the Inspect log
header and records any revisions captured in that header. Pass the rubric as
an explicit task argument and configure one model for the task's grader role
(default `grader`) so both settings are present in the log.

Run the parity command after the separately approved baseline eval has completed:

```powershell
python scripts/run_qa_schema_parity.py --corpus questions/corpus.jsonl --legacy-schema schema.json --v2-schema schemas/conspir/v2_schema.json --eval-log logs/<baseline-qa-run>.eval --model <logged-model-id> --grader <logged-grader-id> --rubric rubric_v2 --output-dir results/schema-parity
```

Save the exported response cache, conversion manifest, and exact-parity report
under `results/schema-parity/`; the source Inspect run log remains under `logs/`.
Do not launch the baseline run until its model/grader settings are pinned and
the contributor explicitly approves it.

## Deviations

<!-- Record any changes after the run. Do not revise the agreed design stages retroactively. -->
