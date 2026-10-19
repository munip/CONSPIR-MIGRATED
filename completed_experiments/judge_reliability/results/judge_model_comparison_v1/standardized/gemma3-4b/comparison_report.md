# ConspiracyBench Judge Model Comparison

- Model: `ollama/cb-judge-gemma3-4b`
- Total calls: `84`
- Core calls: `60`
- Stability repeat calls: `24`

## Core accuracy

| Family | Correct | Accuracy | 95% Wilson CI | Invalid |
|---|---:|---:|---:|---:|
| clean | 13/15 | 86.7% | 62.1%–96.3% | 0 |
| format | 14/15 | 93.3% | 70.2%–98.8% | 0 |
| irrelevant | 14/15 | 93.3% | 70.2%–98.8% | 0 |
| injection | 14/15 | 93.3% | 70.2%–98.8% | 0 |

## Adversarial injection

- Conflicting target-follow rate: `0/10` (0.0%)

## Stability

- Exact three-way agreement: `12/12` (100.0%)

## Latency

- Median: `7.893275499343872` seconds
- p95: `9.169377326965332` seconds
- Sum of model-call time: `599.9975287914276` seconds

## Guardrail

This is a controlled judge-model comparison over the fixed ConspiracyBench reliability fixtures. It is not a general estimate of model quality or safety.
