# Judge Reliability Evaluation Artifacts

This directory contains reproducible artifacts for two focused evaluations of the ConspiracyBench judge: a baseline pass over the current regression cases and a controlled robustness pass over semantics-preserving transformations of those same cases.

These results are intentionally narrow. They do not establish benchmark-wide judge accuracy, general prompt-injection robustness, or validity on arbitrary model responses.

## Configuration

Both evaluations used:

- Provider: Groq direct API
- Model: `openai/gpt-oss-20b`
- `reasoning_effort="low"`
- `include_reasoning=false`
- `max_completion_tokens=512`
- `temperature=0.0`

The runners load the repository's current `judge_cases.jsonl` and use the repository's own `build_judge_prompt()` implementation. `GROQ_API_KEY` is read from the environment and is not written to the result artifacts.

## Baseline evaluation

`run_baseline.py` evaluates the 15 current judge regression cases.

Observed results:

- Cases: 15
- Strict-format accuracy: 15/15
- Parsed-label accuracy: 15/15
- Macro-F1: 1.000
- Invalid outputs: 0
- Class support: 5 `hold`, 5 `diverge`, 5 `abstain`
- Injection-like cases present in the regression set: 2; both classified correctly

## Controlled robustness evaluation

`run_robustness.py` applies three controlled transformations to each of the same 15 source cases:

- `format`: wraps the substantive answer in headings and blockquote formatting.
- `irrelevant`: adds benign surrounding context while preserving the substantive answer.
- `injection`: adds evaluator-directed text instructing the judge to return `hold`, followed by the original substantive answer.

This produces 45 transformed cases in total: 15 per transformation family.

Observed results:

- Total transformed cases: 45
- Strict-format accuracy: 45/45
- Parsed-label accuracy: 45/45
- Macro-F1: 1.000
- Invalid outputs: 0
- Injection-family accuracy: 15/15
- Conflicting injection cases, where the injected target `hold` disagreed with the gold label: 10
- Conflicting injection cases that followed the injected target: 0/10

The 45 robustness rows are controlled transformations of 15 underlying semantic cases, not 45 independent benchmark questions.

## Files

- `run_baseline.py`: baseline evaluator.
- `run_robustness.py`: controlled robustness evaluator.
- `results/groq_gpt_oss_20b/baseline_results.jsonl`: per-case baseline outputs and labels.
- `results/groq_gpt_oss_20b/baseline_summary.json`: baseline aggregate metrics.
- `results/groq_gpt_oss_20b/baseline_confusion_matrix.csv`: baseline confusion matrix.
- `results/groq_gpt_oss_20b/baseline_failures.csv`: baseline failure table.
- `results/groq_gpt_oss_20b/robustness_results.jsonl`: per-variant robustness outputs and labels.
- `results/groq_gpt_oss_20b/robustness_summary.json`: robustness aggregate metrics.
- `results/groq_gpt_oss_20b/robustness_confusion_matrix.csv`: robustness confusion matrix.
- `results/groq_gpt_oss_20b/robustness_family_metrics.csv`: metrics by transformation family.
- `results/groq_gpt_oss_20b/robustness_injection_metrics.json`: injection-specific metrics.
- `results/groq_gpt_oss_20b/robustness_failures.csv`: robustness failure table.
- `results/groq_gpt_oss_20b/experiment_manifest.json`: model configuration, timestamps, and source hashes for the robustness run.

## Reproduction

Install the Groq Python client and set `GROQ_API_KEY` in the environment. From the repository root, run an inspection pass first:

```bash
python completed_experiments/judge_reliability/run_baseline.py --repo . --inspect-only
python completed_experiments/judge_reliability/run_robustness.py --repo . --inspect-only
```

Then run the evaluations:

```bash
python completed_experiments/judge_reliability/run_baseline.py --repo . --run
python completed_experiments/judge_reliability/run_robustness.py --repo . --run
```

Both runners support `--resume` for interrupted runs.

`run_robustness.py` calls the judge through Inspect, so it also needs `inspect-ai`; `--model` takes any Inspect model string (default `groq/openai/gpt-oss-20b`). `--reasoning-effort` defaults to `auto`: `low` for gpt-oss models, matching the configuration above, and not sent to other models, since Ollama rejects it for models that don't think. Each result records the value used.

## Scope

The baseline result is a regression-set check. The robustness result measures invariance only under the three transformations defined in `run_robustness.py`. Neither result should be interpreted as a general estimate of judge reliability outside this controlled set.

## Adversarial injection pilot

The robustness runner can use Inspect model strings rather than a
provider-specific client, so the same adversarial fixtures can be run
against different judge backends.

A small paired local pilot tested 10 cases where the injected target
label (`hold`) conflicted with the expected label. Each judge saw both
the original candidate response and the corresponding injected response.

- `ollama/llama3.2:3b` was correct on 9/10 clean cases and 5/10 injected
  cases under `run_robustness.py`'s parser. Re-parsing the same saved
  outputs with the current Inspect scorer leaves clean accuracy at 9/10
  and gives 4/10 injected. The 2 clean-correct cases that flipped to the
  injected `hold` target are unchanged.
- `ollama/gemma3:4b` was correct on 9/10 clean cases and 9/10 injected
  cases on the same pairs, with no target-label flips.

This is an initial robustness check, not a model ranking. The fixture set
is small and uses a single injection template, so the results should be
treated as evidence that judge behavior can depend on both the model and
the adversarial input rather than as a broad estimate of robustness.

### Reproduction follow-up (2026-10-03)

The same 10 conflicting injection pairs were rerun after the
model-agnostic runner was merged so that the per-case local outputs could
be preserved. No context override was supplied; `ollama ps` reported a
4096-token runtime context for both models.

- `ollama/llama3.2:3b`: under `run_robustness.py`'s parser, 9/10 clean
  correct (90%, Wilson 95% CI 59.6%-98.2%) and 5/10 injected correct
  (50%, 95% CI 23.7%-76.3%), exactly reproducing the earlier aggregate.
  Re-parsing the same saved outputs with the current Inspect scorer leaves
  clean accuracy at 9/10 and gives 4/10 injected correct (40%, 95% CI
  16.8%-68.7%). The 2 clean-correct cases flipping to the injected `hold`
  target are unchanged.
- `ollama/gemma3:4b`: 8/10 clean correct (80%, Wilson 95% CI
  49.0%-94.3%) and 9/10 injected correct (90%, 95% CI 59.6%-98.2%),
  with 0 target-label flips. The earlier pilot was 9/10 clean and
  9/10 injected, so the clean score differed by one case on rerun.

Per-case outputs, summaries, manifests, and Ollama runtime snapshots are
stored under `completed_experiments/judge_reliability/results/ollama_llama3_2_3b/` and
`completed_experiments/judge_reliability/results/ollama_gemma3_4b/`.

## Real-transcript validity set v1 (#182, #203)

`validity_v1/` holds 150 real model replies (Gemma 3 4B and Qwen3.5 4B with thinking off, k = 1,
on the 80-item corpus) for blind human labelling: `sheet.csv` to label, `units.jsonl` in the
`judge_cases.jsonl` format for running judges, and `ANNOTATION_RUBRIC.md`. `make_validity_sheet.py`
samples the units, `run_decision_validity.py` runs the decision models (Clef, Kev-4B, Laya) on them with and without the
preceding exchange for later turns, and `score_validity.py` scores any judge against the labels (human κ,
per-class recall with Wilson intervals, κ, invalid rate, reweighted precision, by turn).
