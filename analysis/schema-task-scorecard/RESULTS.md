# Task-specific scorecard: factual robustness and trait classification

## Claim

For the same pinned model, report Station House factual-robustness performance
and ConspirED six-trait classification performance against their own references,
using task-specific metrics and no combined score or cross-task ranking.

## Unit and metric

Keep task metrics separate; do not aggregate them into a common score.

- **Station House QA:** one unit is each assistant response turn. Report
  opening-turn divergence rate (among answered turns) and abstention rate,
  multi-turn concession and retreat rates, and turn-level trajectories.
- **ConspirED traits:** one unit is each expert-labeled snippet. Report
  six-trait macro-F1, per-trait precision/recall/F1/support, micro-F1,
  six-label exact match, and invalid-output rate. Also report candidate
  precision/recall/F1 using the agreed derived rule (one or more of the six
  traits present), actual and predicted candidate rates, per-trait predicted
  positive rates, average predicted trait count, and the rates of empty and
  all-six-label predictions. Invalid outputs remain in the metrics as no
  predicted traits and count as incorrect for exact match.

An empty trait list means “no traits present,” not an abstention. Trait records
are single-snippet classifications, so QA-style multi-turn trajectories,
concession, and retreat rates do not apply. Identify QA labels as judge outputs
and trait labels as expert annotations.

## Controls

Use the same pinned `hf/Qwen/Qwen2.5-3B-Instruct` revision and deterministic
decoding for both tasks. Keep the task-specific prompts and scoring fixed:
Station House QA uses its existing judge/rubric, while ConspirED traits use the
six expert labels. Report all four context/examples arms from the grouped
five-fold trait study; do not select an arm based on the observed score.
Reuse the single QA run planned for migration parity. Do not tune on the
ConspirED test set, and do not combine or rank scores across tasks.

For context, report a constant always-abstain QA baseline (100% abstention;
divergence rate undefined because it has no answered turns) and the all-six-
present and all-six-absent trait baselines on the same out-of-fold records.
These are within-task references, not cross-task comparators.

## Data

For Station House QA, use the 315-record `questions/corpus.jsonl` artifact
(SHA-256 `22c21ee342796bb1a007be943ddff60ed7d9ece68ed06f381e72411e97fe027b`),
with pending submissions excluded and no other row filters. Reuse its QA
responses from the migration-parity run.

For ConspirED traits, use the 1,305-record `datasets/conspired/train.jsonl`
artifact (SHA-256
`149f80d46018fef0c7ffc0b3be0d9cc6ee33d146b9a7edb9d87d6eb3a7310f45`) and the
accepted five-fold, seed-42 split grouped by 31 source documents. Use the
out-of-fold predictions from all four arms of the trait experiment. Keep the
previously reviewed 669-record test set reserved; it is not part of this
scorecard. The two datasets contain different units and are not paired.

## Noise

Add no reruns beyond the already planned QA baseline and trait experiment.
Use the QA report's metric-specific 95% Wilson intervals for outcome rates and
the trait study's paired 95% document-bootstrap intervals. Keep uncertainty
estimates within each task; do not compare one task's interval or score directly
with the other's.

## Readout

Publish two independent result tables. The Station House table reports judge
outcome rates, turn trajectories, concession/retreat metrics, coverage, and
its intervals. The ConspirED table reports all four arms' trait and candidate
metrics, prediction-rate diagnostics, constant baselines, invalid-output
rates, and its intervals. State each dataset size and reference source.

Do not combine scores, declare a cross-task winner, or interpret differences
between the numeric scales. If a required run or label set is missing, mark
that task's result pending; do not substitute pseudo-labels or infer a score.
This is a descriptive scorecard, not a performance-threshold test. It is
incomplete if either the 315-record QA table or the 1,305-snippet trait table
is missing, if the tables are combined into one score, or if pseudo-labels are
presented as trait gold. Coverage below 315 QA records or 1,305 trait snippets,
or omission of either task table, refutes the report's coverage claim; mark the
scorecard incomplete rather than inferring a result.
