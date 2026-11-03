# ConspirED dominant-trait held-out test evaluation

## Claim

This analysis will characterize how often the frozen Qwen 2.5 3B prompt agrees
with the existing single-label `dominant_trait` annotations on the official
ConspirED test split, and describe the paired difference between snippet-only
and `context1000` prompt arms. It is descriptive and will not claim that context
improves performance or that results generalize beyond this split.

## Unit and metric

One annotated snippet is one unit. Score all 669 official test rows in each
arm. Macro-F1 is the primary summary; also report exact accuracy, the six-class
confusion matrix, valid-label rate, and per-trait precision, recall, F1, and
support. Report the paired context-minus-snippet differences in macro-F1 and
accuracy. “Any-hit” is not a separate metric for this single-label task.

The test supports are 321 `overriding_suspicion`, 193 `nefarious_intent`,
51 `reinterpreting_randomness`, 52 `immune_to_evidence`, 49 `persecuted_victim`,
and 3 `contradictory`. Interpret class-specific results alongside these counts;
the `contradictory` result is especially uncertain.

## Controls

Use the same model revision, frozen prompt, decoding settings, and test items in
both arms. The only arm difference is whether nonblank `context1000` is
included. Include a constant `overriding_suspicion` prediction as a trivial
baseline; choose it from training data only (697 of the 1,244 non-pilot,
context-eligible training rows), never from test labels.

## Data

Use the pinned imported ConspirED data at
`datasets/conspired-context1000-pilot/test.jsonl`. Its SHA-256 is
`9f7c0c472dbd9e060b64af7fb6b8e940a20d073007341515f46d8e5c75432691`.
The import manifest verifies the test output hash, reports zero train/test
source-document overlap, and all 669 test rows have nonblank `context1000`.
The split contains 18 source documents. Use every test row without filtering.
Keep the official test labels closed to prompt changes and model selection.

## Noise

Use greedy decoding and one response per item per arm. Estimate uncertainty for
the paired differences with a paired bootstrap over source documents, not rows.
Use 10,000 bootstrap repetitions with seed 42, sampling 18 document IDs with
replacement and retaining all rows belonging to each sampled document. Keep
the same six classes in every macro-F1 calculation and set undefined
class-specific precision, recall, or F1 to zero within a replicate. There are
only 18 document clusters, so intervals are descriptive and may be wide; the
three `contradictory` examples do not support a stable class-specific estimate.
This analysis has no pass/fail threshold or smallest meaningful effect.

## Readout

Report, for each arm, macro-F1, exact accuracy, valid-label rate, confusion
matrix, and per-trait precision/recall/F1 with support. Report the constant
training-majority baseline on the same test rows. Report paired
context-minus-snippet macro-F1 and accuracy differences with document-bootstrap
intervals, and identify the number of source documents used for resampling.
Do not make a pass/fail decision or a generalization claim.

## How to run

Use `hf/Qwen/Qwen2.5-3B-Instruct` at revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, `--dtype auto`, and greedy
decoding. Keep the current corrected single-label prompt and parser fixed
before test inference. The dedicated full-test runner scores every planned test
ID in both arms and verifies exact coverage. Reproduce the inference run with:

```powershell
python scripts\run_dominant_trait_test.py --dataset datasets\conspired-context1000-pilot\test.jsonl --train-dataset datasets\conspired-context1000-pilot\train.jsonl --model hf/Qwen/Qwen2.5-3B-Instruct --model-revision aa8e72537993ba99e69dfaafa59ed015b17504d1 --dtype auto --seed 42 --bootstrap-repetitions 10000 --max-tasks 1 --max-samples 1 --run-environment "RTX 5060 Laptop GPU (8 GB); dtype=auto (BF16); quantization=none; greedy; max-samples=1; max-tasks=1" --results-dir results\dominant-trait-test-evaluation --log-dir logs\dominant-trait-test-evaluation
```

The runner verifies both imported splits and derives the constant baseline
without using test labels, reconstructing the 10-per-trait pilot selection
from the pinned training data and seed 42. It writes the planned test IDs and
frozen prompt hash before inference, saves per-arm Inspect logs, and requires
complete coverage in both arms.

The machine-readable per-item output and run manifest are
[`report.json`](./report.json) and [`run_manifest.json`](./run_manifest.json).
The inference entry point is
[`run_dominant_trait_test.py`](../../scripts/run_dominant_trait_test.py);
validation, metrics, and document bootstrap are implemented in
[`dominant_trait_test.py`](../../inspect_eval/trait/dominant_trait_test.py).
The shared frozen prompt, label parser, and task builder are in
[`dominant_trait_pilot.py`](../../inspect_eval/trait/dominant_trait_pilot.py).
The pilot sampling protocol is in
[`EXPERIMENT.md`](../../experiment_requests/conspired/dominant-trait-golden-pilot/EXPERIMENT.md);
the source XLSX importer is
[`import_conspired.py`](../../scripts/import_conspired.py).
Raw Inspect logs were retained with the original local run and are not included
in this analysis bundle.

## Deviations

The dedicated runner and report were implemented as specified. Both saved
Inspect logs have successful status and contain the same 669 unique official
test IDs. The frozen prompt SHA-256 is recorded in the run manifest. Inference
used greedy decoding, BF16 (`dtype=auto`), and one sample at a time.

| Arm / baseline | Macro-F1 | Exact accuracy | Valid labels |
|---|---:|---:|---:|
| Snippet only | 0.105 | 0.318 | 669/669 |
| `context1000` | 0.136 | 0.350 | 667/669 |
| Training-majority constant (`overriding_suspicion`) | 0.108 | 0.480 | 669/669 |

The observed paired `context1000` minus snippet-only difference was +0.030
macro-F1 (document-bootstrap 95% interval: +0.013 to +0.047) and +0.031 exact
accuracy (95% interval: +0.002 to +0.054). Treat these as descriptive
comparisons on 18 source documents, not a causal or generalization claim.

Per-trait F1 by support:

| Trait | Support | Snippet-only F1 | `context1000` F1 |
|---|---:|---:|---:|
| `contradictory` | 3 | 0.000 | 0.000 |
| `overriding_suspicion` | 321 | 0.134 | 0.289 |
| `nefarious_intent` | 193 | 0.457 | 0.461 |
| `persecuted_victim` | 49 | 0.040 | 0.032 |
| `immune_to_evidence` | 52 | 0.000 | 0.000 |
| `reinterpreting_randomness` | 51 | 0.000 | 0.032 |

Both arms predicted `nefarious_intent` most often (629/669 snippet-only and
562/669 with context). Neither arm correctly identified any
`immune_to_evidence` examples. The `contradictory` result remains especially
uncertain at support 3. The two context-arm invalid outputs were the
noncanonical misspelling `nefarous_intent`; they remain invalid rather than
being normalized.
