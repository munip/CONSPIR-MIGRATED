# Do the judges improve when given the human labelling rules?

No request was filed for this one: it was run directly by a maintainer on 2026-10-19, and this
file records the question and parameters after the fact.

## Research question

The human labellers worked from `validity_v1/ANNOTATION_RUBRIC.md` (label definitions plus six
precedence rules). The decision judges got three one-line label criteria (#126). Does giving the
judges the same definitions and rules make them agree more with the human labels?

## Parameters

- **Judges:** pplx-decider, Clef, Kev-4B, Laya at the revisions in `decision_judges_v1/protocol.json`.
- **Intervention:** `run_decision_validity.py --criteria rubric`: the verdict question's `criteria`
  become the rubric's label definitions (`RUBRIC_LABELS` in `eval/judge_prompt.py`) and the six rules
  (`RUBRIC_RULES`) are appended to its `instructions`. The state (question, ground truth, the item's
  lists, the reply, the preceding exchange) is byte-identical to v1.
- **Calls:** the same 244 as v1 (150 without context, 94 later turns again with the preceding
  exchange), same order, batch size 1, no truncation, 3 untimed warm-ups.
- **Hardware:** one Modal `H100!` container per judge, the locked v1 environment
  (`modal run decision_judges_v1/modal_app.py::full --criteria rubric`).
- **Scoring:** `score_validity.py` against #364, and against `validity_v1/labels/new_label_agreement.csv`.

## What would answer it

κ, diverge precision and the share labelled diverge, v1 criteria vs rubric criteria, per judge and
variant, plus how many labels change.
