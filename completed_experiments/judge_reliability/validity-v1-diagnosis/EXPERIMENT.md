# Why did the judges do badly on the validity set?

No request was filed for this one: it was run directly by a maintainer on 2026-10-19, and this
file records the question and parameters after the fact.

## Research question

The decision-judge run (#365) and the chat-judge run scored every judge well below what a
benchmark judge needs (best κ 0.58, every usable judge labelling 32–49% of replies diverge where
humans label 23%). Is that the judges, the reference labels (#364), or the instructions the judges
were given? And what would fix it?

## Parameters

- **Data:** the 150 validity units (`validity_v1/units.jsonl`), reference `validity_v1/labels/reference_v1.csv`
  (#364) with its `basis`, `label_a`, `label_b` columns, the three labelling sheets, `key.csv`.
- **Judge outputs:** the published v1 runs only (`results/{clef,kev,laya,pplx}_validity_v1/`,
  `results/chat_judges_validity_v1/`). No new model calls.
- **Analyses:** per-class recall and diverge rate per judge; accuracy on the 117 units both
  labellers agreed on vs the 33 that were adjudicated or resolved by rule; which hold units most
  judges call diverge, read against the item checklists; the human rubric vs the judges' label
  criteria; a sweep of the P(diverge) threshold for pplx-decider and Clef.

## What would answer it

If judges do well where the labellers agreed and badly where they didn't, the reference caps the
judges. If the remaining judge–human disagreements cluster in kinds of reply the rubric doesn't
settle, the fix is a rule decision, not a new judge.
