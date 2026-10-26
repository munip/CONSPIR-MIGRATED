# ConspiracyBench baseline: sensitivity to question categories

**Source:** [ConspiracyBench baseline, October 23, 2026](https://spacesheep.dev/@likelytobelaura/conspiracybench-baseline), using the archived dashboard data in `conspiracybench-baseline.zip`.

**Question:** Does the relative ordering of Granite 4.2 3B and Qwen3.5 4B change when the broad `other` category is excluded?

## Method

The analysis uses 254 multi-turn questions, each evaluated in five conversations. The metric is **ever endorses @5**: a question counts as an endorsement if at least one conversation endorses the conspiracy theory. Questions with any invalid conversation are excluded separately for each model. Abstentions count as non-endorsements.

The script first checks the released item-level results against the published overall numerator and denominator for all five models. It then compares Granite and Qwen on questions eligible for both models, using a paired question-level bootstrap with 20,000 resamples (seed 20261024). Each question, rather than each conversation, is the resampling unit.

## Reproduced endorsement rates

Lower rates indicate fewer questions with at least one endorsed conversation.

| Model | All eligible questions | Excluding `other` | Only `other` |
|---|---:|---:|---:|
| Granite 4.2 3B | 53/253 (20.9%) | 30/159 (18.9%) | 23/94 (24.5%) |
| Qwen3.5 4B | 57/246 (23.2%) | 28/157 (17.8%) | 29/89 (32.6%) |
| Gemma 4 E4B | 69/253 (27.3%) | 34/159 (21.4%) | 35/94 (37.2%) |
| Nemotron 3 Nano 4B | 99/230 (43.0%) | 56/144 (38.9%) | 43/86 (50.0%) |
| K2 Horizon 3.7B | 116/243 (47.7%) | 62/154 (40.3%) | 54/89 (60.7%) |

All five overall rates match the published baseline. The `other` category contains 94 of 254 multi-turn questions (37.0%).

## Granite–Qwen comparison on matched questions

Differences are Qwen minus Granite, in percentage points; positive values indicate more endorsements by Qwen.

| Question subset | Matched questions | Qwen endorsements | Granite endorsements | Difference | 95% bootstrap interval |
|---|---:|---:|---:|---:|---:|
| All | 245 | 57 | 49 | +3.3 pp | -2.4 to +9.0 pp |
| Excluding `other` | 156 | 28 | 29 | -0.6 pp | -7.1 to +6.4 pp |
| Only `other` | 89 | 29 | 20 | +10.1 pp | -1.1 to +21.3 pp |

## Interpretation

Granite has the lower overall endorsement rate, while Qwen has the lower rate outside `other`. This change in descriptive ordering also appears in the matched-question comparison. However, all three bootstrap intervals include zero, so the data do not establish a reliable difference between the two models in these subsets.

This is an exploratory check of category composition, not a causal test. The `other` category groups different topics, and eligibility differs by model. The analysis also does not independently reassess the judge's decisions.

## Reproduction

With `conspiracybench-baseline.zip` in the same directory:

```bash
python analyze_conspiracybench_category_sensitivity.py conspiracybench-baseline.zip
```

Only the Python standard library is required. The script checks that all five overall numerators and denominators agree with the published dashboard.
