# ConspiracyBench Judge Model Comparison

- Model: `ollama/cb-judge-qwen3-4b-instruct-2507`
- Total calls: `84`
- Core calls: `60`
- Stability repeat calls: `24`

## Core accuracy

| Family | Correct | Accuracy | 95% Wilson CI | Invalid |
|---|---:|---:|---:|---:|
| clean | 13/15 | 86.7% | 62.1%–96.3% | 0 |
| format | 13/15 | 86.7% | 62.1%–96.3% | 0 |
| irrelevant | 11/15 | 73.3% | 48.0%–89.1% | 0 |
| injection | 11/15 | 73.3% | 48.0%–89.1% | 0 |

## Adversarial injection

- Conflicting target-follow rate: `2/10` (20.0%)

## Stability

- Exact three-way agreement: `12/12` (100.0%)

## Latency

- Median: `2.2669804096221924` seconds
- p95: `18.28383231163025` seconds
- Sum of model-call time: `419.3840219974518` seconds

## Guardrail

This is a controlled judge-model comparison over the fixed ConspiracyBench reliability fixtures. It is not a general estimate of model quality or safety.
