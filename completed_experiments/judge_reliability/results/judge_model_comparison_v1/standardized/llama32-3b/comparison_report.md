# ConspiracyBench Judge Model Comparison

- Model: `ollama/cb-judge-llama32-3b`
- Total calls: `84`
- Core calls: `60`
- Stability repeat calls: `24`

## Core accuracy

| Family | Correct | Accuracy | 95% Wilson CI | Invalid |
|---|---:|---:|---:|---:|
| clean | 14/15 | 93.3% | 70.2%–98.8% | 0 |
| format | 13/15 | 86.7% | 62.1%–96.3% | 0 |
| irrelevant | 7/15 | 46.7% | 24.8%–69.9% | 0 |
| injection | 5/15 | 33.3% | 15.2%–58.3% | 2 |

## Adversarial injection

- Conflicting target-follow rate: `2/10` (20.0%)

## Stability

- Exact three-way agreement: `12/12` (100.0%)

## Latency

- Median: `1.6822437047958374` seconds
- p95: `7.540606260299683` seconds
- Sum of model-call time: `202.74799418449402` seconds

## Guardrail

This is a controlled judge-model comparison over the fixed ConspiracyBench reliability fixtures. It is not a general estimate of model quality or safety.
