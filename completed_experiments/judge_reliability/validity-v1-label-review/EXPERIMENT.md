# Which validity-set labels should change?

No request was filed for this one: it was run directly by a maintainer on 2026-10-19, and this
file records the question and parameters after the fact.

## Research question

The diagnosis (`../validity-v1-diagnosis/`) found that every judge does badly on the 33 units
without two agreeing labels, and that several resolved labels conflict with their item's own
checklist. Which of those weak labels should change, and to what?

## Parameters

- **Units reviewed:** the 20 weakest of the 33 non-agreed units: 12 where 5–6 of the 6 v1 judges
  then compared disagreed with the resolved label and one human had agreed with them, plus 8 where
  4 of 6 disagreed.
- **Reviewer:** a model (Claude Opus 5.5), reading each reply against its question, ground truth,
  `diverges_if` / `does_not_diverge_if` lists and the labelling rubric, and recording the checklist
  entry or rubric rule that decided it. These are proposals, not human labels.
- **Output:** `validity_v1/labels/proposed_v1.1_review.csv` (one row per reviewed unit) and
  `validity_v1/labels/new_label_agreement.csv` (all 150 units with high- and medium-confidence
  changes applied; low-confidence and rule-dependent units left pending).

## What would answer it

A per-unit proposal, with confidence and the deciding rule, that a blind third labeller can check.
