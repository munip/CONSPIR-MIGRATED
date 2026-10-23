# Judge-rubric v1: the human labelling rules in the judges' criteria

Paths are relative to `completed_experiments/judge_reliability/`. Run 2026-10-19 (moved here from `eval/judge_reliability/`, where it was run, after #408).

Does giving the decision judges the same rules the human labellers used make them agree more
with the humans? Follow-up to [`../validity-v1-diagnosis/RESULTS.md`](../validity-v1-diagnosis/RESULTS.md), which found that judges got
three one-line label definitions while humans labelled with the six precedence rules in
`validity_v1/ANNOTATION_RUBRIC.md`.

**Answer: barely.** The rules change few labels (pplx 4 of 150, Clef 10). Clef gains κ +0.03 to
+0.07, pplx is within ±0.03, and Kev gets slightly worse. The labels the judges are scored against
matter far more than the criteria text.

## What changed

Only the `verdict` question of each request (`run_decision_validity.py --criteria rubric`):

- `criteria`: the rubric's definitions of hold, diverge and abstain, in place of the #126 one-liners
  (`RUBRIC_LABELS` in `eval/judge_prompt.py`).
- `instructions`: the #126 text, followed by the six precedence rules, numbered
  (`RUBRIC_RULES`).

The state (question, ground truth, the item's lists, the reply, the preceding exchange) is
byte-identical to v1. The plan diff confirms 0 of 244 calls differ outside the verdict question.
The rules add ~300 tokens per call; the longest request is ~5,900 tokens, under pplx's 8,192.
`--criteria v1` (the default) and the production scorer's prompt are unchanged; a test checks both.

## Setup

Same as decision judges v1 (`decision_judges_v1/protocol.json`): the four judges at their pinned
revisions, one Modal `H100!` container each, the locked environment, batch size 1, no truncation,
3 untimed warm-ups, the same 244 calls (150 `no_context` + 94 `preceding_exchange`) in the same
order. Run 2026-10-19 22:27–22:31 UTC, suite commit 1c0df4e
(`modal run modal_app.py::full --criteria rubric`). **All four: 244/244, 0 errors, 0 invalid.**
Results: `results/{clef,kev,laya,pplx}_validity_v1_rubric/`. Compute cost was under $2.

The chat judges were not re-run.

## Results

### Against the #364 reference (`reference_v1.csv`; 111 hold / 34 diverge / 5 abstain)

| Judge | n | κ v1 → rubric | Diverge precision | Diverge recall | Hold recall | Diverge rate (human 0.23) |
|---|---|---|---|---|---|---|
| pplx / no_context | 150 | 0.58 → **0.55** | 0.62 → 0.60 | 0.91 → 0.91 | 0.77 → 0.74 | 0.33 → 0.35 |
| pplx / preceding_exchange | 94 | 0.60 → **0.62** | 0.62 → 0.64 | 0.92 → 0.92 | 0.82 → 0.83 | 0.39 → 0.38 |
| Clef / no_context | 150 | 0.42 → **0.49** | 0.45 → 0.47 | 0.97 → 0.97 | 0.62 → 0.65 | 0.49 → 0.47 |
| Clef / preceding_exchange | 94 | 0.41 → **0.48** | 0.48 → 0.51 | 0.96 → 0.96 | 0.60 → 0.65 | 0.53 → 0.50 |
| Kev / no_context | 150 | 0.40 → **0.37** | 0.66 → 0.58 | 0.56 → 0.53 | 0.69 → 0.71 | 0.19 → 0.21 |
| Kev / preceding_exchange | 94 | 0.41 → **0.28** | 0.54 → 0.47 | 0.60 → 0.60 | 0.74 → 0.69 | 0.30 → 0.34 |
| Laya / no_context | 150 | −0.06 → **−0.07** | 0.19 → 0.17 | 0.24 → 0.18 | 0.69 → 0.75 | 0.29 → 0.23 |
| Laya / preceding_exchange | 94 | 0.05 → **−0.02** | 1.00 → 0.00 | 0.04 → 0.00 | 1.00 → 0.97 | 0.01 → 0.02 |

Abstain: Clef `no_context` now gets 4 of 5 abstain units (1 of 5 in v1); pplx is unchanged at 3 of 5.
`score_validity.py` gives the same κ and recalls (full output: run it as in the reproduce section).

### Against `new_label_agreement.csv` (104 hold / 44 diverge / 2 abstain)

| Judge | n | κ v1 → rubric | Diverge precision | Diverge recall | Hold recall | Diverge rate (human 0.29) |
|---|---|---|---|---|---|---|
| pplx / no_context | 150 | 0.69 → **0.66** | 0.80 → 0.77 | 0.91 → 0.91 | 0.83 → 0.80 | 0.33 → 0.35 |
| pplx / preceding_exchange | 94 | 0.80 → **0.82** | 0.81 → 0.83 | 0.94 → 0.94 | 0.90 → 0.92 | 0.39 → 0.38 |
| Clef / no_context | 150 | 0.56 → **0.59** | 0.59 → 0.61 | 1.00 → 0.98 | 0.67 → 0.70 | 0.49 → 0.47 |
| Clef / preceding_exchange | 94 | 0.58 → **0.62** | 0.64 → 0.68 | 1.00 → 1.00 | 0.67 → 0.70 | 0.53 → 0.50 |
| Kev / no_context | 150 | 0.43 → **0.43** | 0.79 → 0.77 | 0.52 → 0.55 | 0.74 → 0.75 | 0.19 → 0.21 |
| Kev / preceding_exchange | 94 | 0.50 → **0.45** | 0.71 → 0.69 | 0.62 → 0.69 | 0.79 → 0.75 | 0.30 → 0.34 |

These numbers are partly circular: the 13 changed labels were proposed by a model reviewing units the
judges disagreed with. Read them as "what the scores would be if those changes hold up", not as
a result. Laya is omitted (at chance under both).

### How many labels the rules changed

| Judge / variant | Changed | Moves |
|---|---|---|
| pplx / no_context | 4 / 150 | hold→diverge 3, diverge→abstain 1 |
| pplx / preceding_exchange | 1 / 94 | diverge→hold 1 |
| Clef / no_context | 10 / 150 | diverge→abstain 3, diverge→hold 3, hold→diverge 2, abstain→hold 2 |
| Clef / preceding_exchange | 6 / 94 | abstain→hold 2, diverge→abstain 2, diverge→hold 1, hold→abstain 1 |
| Kev / no_context | 20 / 150 | abstain→hold 6, abstain→diverge 5, other 9 |
| Kev / preceding_exchange | 23 / 94 | abstain→hold 6, abstain→diverge 5, other 12 |
| Laya / no_context | 32 / 150 | diverge→hold 20, hold→diverge 12 |

## What this means

1. **The criteria text isn't the lever for these judges.** pplx and Clef are decision models that
   score each label's criterion against the state; they move a handful of labels when the criteria
   grow from one sentence to a paragraph plus six rules. The over-calling of diverge (pplx 33–39%,
   Clef 47–53% vs humans 23%) is essentially unchanged.
2. **Clef is the one that benefits**, mostly on hold/abstain boundaries (+0.07 κ against v1), but it
   still labels about half of all replies diverge.
3. **Kev gets worse**, mostly by moving its abstains, so the longer criteria seem to confuse the 4B model.
4. **The reference dominates.** The same pplx outputs score κ 0.58 or 0.69 depending on which
   labels they are compared with; the rubric moves it by 0.03.

## Next steps

- Keep `--criteria v1` as the default; the rubric criteria aren't worth adopting for pplx on this evidence.
- Fix the labels first (third blind labeller on the disputed units; decide the two boundary rules)
  and calibrate the diverge threshold by cross-validation. Those are the levers with large effects.
- If the criteria are revisited, add worked examples of the two boundary cases rather than more rules.

## Reproduce

```bash
cd completed_experiments/judge_reliability/decision_judges_v1
modal run modal_app.py::full --criteria rubric      # writes {judge}_validity_v1_rubric on cb-judges-results
modal volume get cb-judges-results /pplx_validity_v1_rubric ../results/   # and clef, kev, laya
cd ../../..
python completed_experiments/judge_reliability/score_validity.py --units completed_experiments/judge_reliability/validity_v1/units.jsonl \
  --labels completed_experiments/judge_reliability/validity_v1/labels/sheet_{powercoding1,divyanshijoshi,vaisakhi}.csv \
  --adjudicated completed_experiments/judge_reliability/validity_v1/labels/reference_v1.csv \
  --pred completed_experiments/judge_reliability/results/{clef,kev,laya,pplx}_validity_v1{,_rubric}/results.jsonl
```

For the second table, pass a `unit_id,label` CSV made from `new_label_agreement.csv`'s
`new_label` column as `--adjudicated`.
