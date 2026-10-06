# ConspiracyBench Judge Model Comparison

- Model: `clef/Cloudflare/clef@2f3de3dd85f3`
- Total calls: `84`
- Core calls: `60`
- Stability repeat calls: `24`

## Core accuracy

| Family | Correct | Accuracy | 95% Wilson CI | Invalid |
|---|---:|---:|---:|---:|
| clean | 15/15 | 100.0% | 79.6%–100.0% | 0 |
| format | 15/15 | 100.0% | 79.6%–100.0% | 0 |
| irrelevant | 15/15 | 100.0% | 79.6%–100.0% | 0 |
| injection | 15/15 | 100.0% | 79.6%–100.0% | 0 |

## Adversarial injection

- Conflicting target-follow rate: `0/10` (0.0%)

## Stability

- Exact three-way agreement: `12/12` (100.0%)

## Latency

- Median: `0.43206058582291007` seconds
- p95: `0.4578863251954317` seconds
- Sum of model-call time: `36.95165099296719` seconds

## Guardrail

This is a controlled judge-model comparison over the fixed ConspiracyBench reliability fixtures. It is not a general estimate of model quality or safety.
