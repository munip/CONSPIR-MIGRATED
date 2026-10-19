# ConspiracyBench Judge Context Stress

- Model: `ollama/cb-judge-llama32-3b`
- Calls: `9`
- Suite: `context-stress-1`

## Accuracy by condition

| Condition | Correct | Accuracy | Invalid | Median prompt chars | Provider-reported median total tokens |
|---|---:|---:|---:|---:|---:|
| baseline | 3/3 | 100.0% | 0 | 1932 | 385 |
| ground_truth_early | 3/3 | 100.0% | 0 | 70334 | 14381 |
| ground_truth_late | 2/3 | 66.7% | 0 | 70334 | 14381 |

## Interpretation guardrail

This stress test changes the amount and position of neutral context inside the ground-truth field while keeping the candidate answer and expected label fixed. Provider-reported total-token counts are retained only as local calibration diagnostics and should not be treated as exact cross-provider measurements without separate validation.
