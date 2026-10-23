# Results: judge reliability rerun on the final labels

Run 2026-10-22, 19:55–19:59 UTC: four judges, 244 calls each, 0 errors. All scores are against the
final labels (`reference_v1.2.csv`: 112 hold, 33 diverge, 5 abstain), which are treated as ground
truth. Combined explorer with the Experiment 005 injection study:
https://spacesheep.dev/@likelytobelaura/judge-trust

## Deployed setup (endorsement rules, context on later turns), all 150 replies

| Judge | Accuracy (95% CI) | Cohen's κ (95% CI) | Diverge recall / precision | Predicted hold / diverge / abstain |
|---|---|---|---|---|
| pplx-decider | 90.7% (85–95) | 0.755 (0.62–0.86) | 85% / 80% | 114 / 35 / 1 |
| Clef | 89.3% (84–94) | 0.730 (0.60–0.84) | 88% / 74% | 110 / 39 / 1 |
| Kev-4B | 79.3% (73–86) | 0.390 (0.21–0.55) | 42% / 61% | 126 / 23 / 1 |
| Laya | 73.3% (66–80) | −0.019 (−0.05–0.00) | 0% / 0% | 148 / 1 / 1 |

Always answering "hold" would score 74.7%, so Laya's accuracy is the majority-class rate.

- **pplx-decider and Clef are the usable judges**, with κ about 0.73–0.76 and diverge recall of
  85–88%.
- **Kev-4B misses most diverges.** It also does worse with context: κ is 0.52 without the preceding
  exchange and 0.39 with it.
- **Laya labels almost everything "hold"** and is not informative.

## Errors (rows = final label, columns = judge)

| | pplx: hold / diverge / abstain | Clef: hold / diverge / abstain |
|---|---|---|
| hold (112) | 107 / 5 / 0 | 104 / 8 / 0 |
| diverge (33) | 5 / 28 / 0 | 4 / 29 / 0 |
| abstain (5) | 2 / 2 / 1 | 2 / 2 / 1 |

## Reproducibility

All four judges reproduced the 2026-10-19 rubric_v2 run exactly: 244/244 identical choices each,
with a largest probability difference of 0.0.

## Limits

- 150 replies from one validity set. Abstain has only 5 examples.
- The endorsement rules were written while reading these same 150 replies, so these scores are
  in-sample for the rules.
