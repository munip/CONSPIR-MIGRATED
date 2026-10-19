# Clef judge experiment

## Research question

Can Cloudflare Clef serve as a stronger judge for ConspiracyBench than the currently tested local judge candidates when evaluated under the same judge-reliability protocol?

The primary candidate for this experiment is the full Clef model linked in discussion #123. Clef-Flash may be used for an initial feasibility run if the available compute cannot support the full model, but results from the two variants should be reported separately.

## Motivation

The current judge-model comparison in #122 evaluates Llama 3.2 3B, Gemma 3 4B, and Qwen3 4B Instruct 2507.

Qwen3 4B Instruct 2507 is currently the strongest tested local candidate overall, while Gemma 3 4B performs best on the synthetic robustness suite. The comparison does not yet include a substantially larger judge designed around typed decisions.

Clef is interesting because its decision-oriented interface may avoid some of the free-form parsing and reply-budget failure modes seen with ordinary generative judges.

This experiment is intended as an additional candidate evaluation. It does not replace or rewrite the existing #122 results.

## Comparison protocol

Use the same cases and scoring definitions as the standardized comparison introduced in #122.

The existing standardized plan contains 84 calls per candidate:

- 15 clean cases
- 15 formatting-robustness cases
- 15 irrelevant-context cases
- 15 prompt-injection cases
- 24 repeated calls used for stability checking

The Clef adapter should expose the same three benchmark decisions:

- `hold`
- `diverge`
- `abstain`

If the Clef runtime exposes confidence or probability values for the available choices, preserve those values in the raw output for analysis. They should not replace the benchmark label used by the existing scorer.

## Phase A: adapter preflight

### Fixed preflight selection and implementation

The six calls are fixed before Clef is executed: clean `judge-001` (`hold`), `judge-003` (`diverge`), and `judge-005` (`abstain`), plus the corresponding prompt-injection variants of those same three cases. The injection target is `hold`, so `judge-003` and `judge-005` are conflicting-target cases.

The adapter uses Clef's native typed `choice` interface with exactly three benchmark choices: `hold`, `diverge`, and `abstain`. The candidate response is treated as data, and Clef's selected choice is used directly as the benchmark label. Confidence and per-choice probabilities are retained in the raw result.

The executable runner is `completed_experiments/judge_reliability/run_clef_preflight.py`, with model-independent tests in `completed_experiments/judge_reliability/test_clef_preflight.py`.

The full-model preflight is pinned to `Cloudflare/clef` revision `2f3de3dd85f379784083b0814d997ab627200f0c`, using `bfloat16`.

Run:

`python completed_experiments/judge_reliability/run_clef_preflight.py --repo . --output-dir completed_experiments/judge_reliability/results/clef_preflight_v1 --model-id Cloudflare/clef --revision 2f3de3dd85f379784083b0814d997ab627200f0c --device cuda --dtype bfloat16 --run`

Before requesting a full benchmark run, execute a fixed six-call preflight using existing cases only:

- one clean case with expected label `hold`;
- one clean case with expected label `diverge`;
- one clean case with expected label `abstain`;
- the corresponding prompt-injection variant for each of those same three source cases.

Select and record these six case IDs before executing Clef. Do not change the preflight cases in response to model outputs.

Proceed to the full 84-call comparison only if all six calls complete without adapter/runtime failure and each produces exactly one valid mapped benchmark decision.

The preflight should verify:

1. the model accepts the rendered judge input;
2. the three benchmark choices can be represented without changing their meaning;
3. every completed call produces one valid benchmark decision;
4. any confidence or probability output can be recorded reproducibly;
5. model identity, revision, runtime configuration, and latency are captured;
6. failures are reported rather than silently retried into a valid result.

If the model cannot reliably produce a valid mapped decision under the fixed adapter, record this as a preflight failure and stop rather than modifying the benchmark protocol until it succeeds.

## Phase B: standardized comparison

If the preflight succeeds, run the existing 84-call standardized suite without changing the cases or expected labels.

Report at least:

- overall core accuracy;
- clean accuracy;
- formatting-robustness accuracy;
- irrelevant-context accuracy;
- prompt-injection accuracy;
- conflicting injection target-follow rate;
- invalid-output rate;
- repeat stability;
- median latency;
- model/runtime provenance.

Where available, also retain Clef's decision confidence or probability values in the raw artifacts.

The purpose of this phase is a direct comparison with the Llama, Gemma, and Qwen results already reported in #122.

## Phase C: follow-up panels

If Clef is competitive in the standardized suite, run the additional #122 evaluation panels:

1. the 20-response real-transcript agreement panel with the existing manually verified reference labels;
2. the nine-call near-context-limit stress panel.

Do not use these panels to alter the labels or tune the adapter.

The real-transcript panel should report agreement, supported-class macro-F1, and invalid outputs.

The context panel should report baseline, ground-truth-early, and ground-truth-late accuracy using the same source cases and context construction as #122.

## Decision rule

Use the same evidence hierarchy already documented for the current judge comparison:

1. real-transcript agreement;
2. worst-case performance across the standardized core robustness families;
3. injection target-follow and invalid-output behavior;
4. near-context-limit robustness;
5. latency as an operational tie-breaker.

No final primary-judge recommendation should be made from the preflight alone.

## Compute request

The preferred run is the full Clef model linked in #123.

Neither machine currently available to the contributor has an NVIDIA/CUDA GPU. The personal machine has approximately 8 GB system RAM, and the school machine has approximately 32 GB system RAM with Intel integrated graphics only.

Project GPU compute is therefore requested for the model run.

The first use of the compute should be the Phase A preflight. The full 84-call suite should run only if that preflight succeeds.

If the available project GPU cannot support the full Clef model, Clef-Flash can be considered as a separately identified feasibility candidate rather than silently substituting it for full Clef.

Avoid quantizing the initial full-model comparison unless required by the available hardware. If quantization or another material runtime change is necessary, record it explicitly in the run metadata and report.

## Reproducibility

Record:

- exact model identifier and revision;
- runtime/library versions;
- hardware used;
- adapter configuration;
- benchmark/suite commit;
- raw per-case outputs;
- any confidence/probability outputs exposed by Clef;
- latency;
- failures and retries.

The standardized comparison should use the exact #122 cases and expected labels so that differences are attributable to the candidate judge rather than a changed test set.

## Related work

- #113: Recommend Judge Models task thread
- #122: standardized judge-model comparison harness and current results
- #123: Clef/Jev judge-model discussion
- #125: compute-request workflow