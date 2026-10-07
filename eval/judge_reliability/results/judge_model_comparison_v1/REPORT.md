# Judge model comparison

## Recommendation

Among the three fully tested local candidates, Qwen3 4B Instruct 2507 is the strongest overall candidate and the recommended local fallback.

It has the strongest agreement with the manually verified real-transcript panel and the strongest near-context-limit result. Gemma 3 4B performs best on the synthetic robustness suite, but its real-transcript agreement is substantially worse. Llama 3.2 3B is fastest on the standardized suite but is not competitive on robustness or transcript agreement.

The overall primary-judge selection is not considered finished. A stronger model such as gpt-oss-20b should be run through the same standardized harness before choosing the final primary judge.

## Standardized robustness suite

Each model received the same 84-call task plan:

- 15 clean cases
- 15 formatting-robustness cases
- 15 irrelevant-context cases
- 15 prompt-injection cases
- 24 additional repeated calls for deterministic-repeat checking

Temperature was fixed at 0.

| Model | Overall | Clean | Format | Irrelevant | Injection | Injection target-follow | Invalid | Repeat stability | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama 3.2 3B | 39/60 (65.0%) | 14/15 | 13/15 | 7/15 | 5/15 | 2/10 | 2 | 12/12 | 1.68 s |
| Gemma 3 4B | 55/60 (91.7%) | 13/15 | 14/15 | 14/15 | 14/15 | 0/10 | 0 | 12/12 | 7.89 s |
| Qwen3 4B Instruct 2507 | 48/60 (80.0%) | 13/15 | 13/15 | 11/15 | 11/15 | 2/10 | 0 | 12/12 | 2.27 s |

Wilson intervals for the standardized families are stored in each model's `comparison_summary.json`.

Llama's low injection accuracy should not be interpreted as simple instruction-following: only 2 of the 10 conflicting-target injection cases actually followed the injected target. The remaining failures are more consistent with destabilization or confusion.

## Real-transcript agreement

A separate panel was generated from six real solver conversations with scoring disabled during generation. The final panel contains 20 response-level grading units.

A single human reviewer manually checked the 20 reference labels before candidate-judge predictions were generated.

The production Inspect scorer uses no transcript context on an opening response and preceding history on later pressure turns. An initial analysis passed transcript context to the six opening responses; those 18 model calls were rerun with `transcript=None`, and the corrected Inspect-parity results below are the canonical results.

Reference support is 18 hold, 2 diverge, and 0 abstain.

| Model | Agreement | Supported-class macro-F1 | Invalid |
| --- | ---: | ---: | ---: |
| Qwen3 4B Instruct 2507 | 18/20 (90.0%) | 0.804 | 0 |
| Llama 3.2 3B | 7/20 (35.0%) | 0.317 | 1 |
| Gemma 3 4B | 6/20 (30.0%) | 0.293 | 0 |

The panel is small and class-imbalanced, and it contains no abstain reference examples. It therefore does not measure real-transcript abstain agreement.

The solver used to generate these transcripts was Qwen3 4B Instruct 2507, so the Qwen judge result is same-family with the solver. That is a limitation of this pilot and should be tested with additional solver families in future work.

## Near-context-limit stress test

Three balanced source cases were tested as baseline, with long neutral context before the ground truth, and with long neutral context after the ground truth.

| Model | Baseline | Ground truth early | Ground truth late | Total |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 4B Instruct 2507 | 3/3 | 3/3 | 3/3 | 9/9 |
| Llama 3.2 3B | 3/3 | 3/3 | 2/3 | 8/9 |
| Gemma 3 4B | 3/3 | 1/3 | 1/3 | 5/9 |

This is a small context stress test, not a realistic long multi-turn benchmark, so it is treated as supporting evidence rather than a standalone selection criterion.

## Qwen3.5 4B

Qwen3.5 4B did not complete the preflight protocol: on the smoke test it exhausted the fixed 512-token judge reply budget without emitting a final label. It is therefore recorded as a preflight failure rather than being silently omitted or treated as a completed 84-call candidate.

## Current decision

Using the fixed decision order, Qwen3 4B Instruct 2507 is the strongest of the three fully evaluated local candidates.

Gemma remains an important robustness reference because it performs best on the synthetic robustness suite, especially irrelevant-context and injection cases, but its poor transcript agreement prevents recommending it as the primary local judge.

Llama is operationally faster but performs substantially worse on irrelevant-context robustness, injection robustness, and real-transcript agreement.

The next useful comparison is a stronger judge such as gpt-oss-20b run under the exact same harness. Until that comparison is available, Qwen3 4B Instruct 2507 is the recommended local fallback rather than a final recommendation for the project's overall primary judge.

## Reproducibility

`run_model_comparison.py` generates the standardized task plan, raw JSONL predictions, summaries, reports, preflight metadata, and manifests.

The standardized runs in this directory were originally generated from suite commit `e41e6de`. That comparison-suite commit was subsequently cherry-picked unchanged onto current main for this follow-up branch.

Canonical raw artifacts are included under:

- `standardized/`
- `transcript_agreement/`
- `context_stress/`

Temporary progress files and duplicate partial-result files are intentionally excluded.
