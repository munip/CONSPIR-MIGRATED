# ConspiracyBench Judge Context Stress

- Model: `ollama/cb-judge-qwen3-4b-instruct-2507`
- Calls: `9`
- Suite: `context-stress-1`

## Accuracy by condition

| Condition | Correct | Accuracy | Invalid | Median prompt chars | Provider-reported median total tokens |
|---|---:|---:|---:|---:|---:|
| baseline | 3/3 | 100.0% | 0 | 1932 | 368 |
| ground_truth_early | 3/3 | 100.0% | 0 | 63934 | 14466 |
| ground_truth_late | 3/3 | 100.0% | 0 | 63934 | 14466 |

## Interpretation guardrail

This stress test changes the amount and position of neutral context inside the ground-truth field while keeping the candidate answer and expected label fixed. Provider-reported total-token counts are retained only as local calibration diagnostics and should not be treated as exact cross-provider measurements without separate validation.
