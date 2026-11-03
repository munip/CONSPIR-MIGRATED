# ConspirED dominant-trait golden-set pilot

## Claim

A small exploratory pilot will use a direct prompt to ask Qwen 2.5 3B to predict
one dominant trait for each source-annotated snippet. The pilot will characterize
output-format failures and trait-confusion patterns; it will not support a
performance or generalization claim.

## Unit and metric

One annotated ConspirED snippet is one unit; the pilot contains 60 snippets.
The reference is the existing `dominant_trait` label derived from the upstream
one-hot annotation. The model must return exactly one of the six dominant
traits. The one-hot bit order is `contradictory`,
`overriding_suspicion`, `nefarious_intent`, `persecuted_victim`,
`immune_to_evidence`, `reinterpreting_randomness`.

Report the valid-single-label rate, exact accuracy, the six-class confusion
matrix, and per-class correct/incorrect counts. Treat all results as
descriptive pilot diagnostics, not evidence of general performance or
generalization.

## Controls

Use a constant majority-trait prediction as a trivial baseline; determine the
majority class from development data, not from held-out pilot items. After
removing the balanced 60-item pilot, the remaining 1,244 context-eligible
training items have `overriding_suspicion` as the majority class (697/1,244).
On the balanced pilot, always predicting that class scores 10/60.

The matched model comparison uses the same direct dominant-trait prompt and
changes only whether `context1000` is supplied: annotated snippet only versus
`context1000` plus annotated snippet.

## Data

Use only the official ConspirED training split and the existing
`dominant_trait` labels. The source revision is recorded as `main`; pin the
local source files by SHA-256: training
`285fdf35b507c8d4306661cc00f1f987da018f237e4c332724935231f8981dfb` and test
`a4b52c52eee870ff18ab21d2f9f421c69bd09f6e816a5179d5272e71c0e5fa8a`. The
official train/test splits contain 1,305 and 669 records, respectively, and
have no overlapping source documents. Re-import the source with
`context_column=context1000` into a new directory. The resulting training
JSONL SHA-256 is
`72352c12965f48d5dfa0d0a403f98ad81f7edfdff05d437021719adec3f478d4` and the
test JSONL SHA-256 is
`9f7c0c472dbd9e060b64af7fb6b8e940a20d073007341515f46d8e5c75432691`; verify
both against the generated manifest.
The [upstream README](https://github.com/UKPLab/conspired) defines
`dominant_consol_trait` as the “most salient/dominant trait in the snippet” and
lists `single_dominant_one_hot_dm` as the one-hot dominant-trait vector. The
repository importer maps that vector to `dominant_trait`; use it as the
reference and record the three train rows where `dominant_consol_trait`
conflicts with the one-hot label.

Require nonblank snippet text, a source document ID, a valid dominant-trait
label, and nonblank `context1000`; exclude the one training record missing
context from both arms. This leaves 1,304 eligible training records. Draw a
class-balanced diagnostic sample from those records and leave the official
test split untouched. Record selected IDs and source documents; favor distinct
documents where feasible and report repeats.

## Noise

Select 60 items deterministically with seed 42, balanced at 10 items per
dominant-trait class. Favor distinct source documents where feasible and report
any repeated documents. Run each item once in each context arm with greedy
decoding and pair results by item. Summarize uncertainty descriptively by
source document; the pilot has no pass/fail threshold and supports no
performance or generalization claim. The prepared sample covers all 31 source
documents; 28 documents contribute multiple snippets, with at most three
snippets from any one document.

## Readout

For each arm, report the valid-single-label rate, exact accuracy, the
six-class confusion matrix, and per-class support/correct/incorrect counts.
Report the majority-class baseline on the same 60 items and the paired counts
of examples helped or hurt by adding `context1000`, with representative
disagreements for manual inspection.

The pilot succeeds only if both arms complete and capture every planned item;
an arm failing or missing an item refutes this plumbing-success criterion.
No model-performance claim is made, so no accuracy result supports or refutes
general performance. Do not use these results to claim that either arm is
better or that performance generalizes. Because the pilot may inform prompt
changes, do not reuse it as a confirmatory gold set.

## How to run

Use `hf/Qwen/Qwen2.5-3B-Instruct` at revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, with `--dtype auto` (the checkpoint
declares BF16), greedy decoding (`do_sample=False`), and sample concurrency 1.
The Inspect Hugging Face provider must not receive `temperature=0`; Transformers
rejects it for generation when greedy decoding is used.

Re-import the source files into a new output directory with:

```powershell
python scripts\import_conspired.py --train-xlsx datasets\source\conspired\context_training.xlsx --test-xlsx datasets\source\conspired\context_testing.xlsx --output-dir datasets\conspired-context1000-pilot --source-revision main --context-column context1000
```

Verify the source and output hashes and the manifest before inference. Do not
overwrite the existing `datasets/conspired/` import. The existing
`scripts/run_trait_prompt_experiment.py` expects multi-label predictions and is
not a valid runner for this single-label pilot. Use the dedicated
`scripts/run_dominant_trait_pilot.py` runner. First prepare and inspect the
deterministic sample manifest without model inference:

```powershell
python scripts\run_dominant_trait_pilot.py --dataset datasets\conspired-context1000-pilot\train.jsonl --run-environment "RTX 5060 Laptop GPU (8 GB); dtype=auto (BF16); quantization=none; greedy; max-samples=1; max-tasks=1" --prepare-only --results-dir results\dominant-trait-context-pilot-bf16 --log-dir logs\dominant-trait-context-pilot-bf16
```

After approving the sample IDs, run the two paired arms with fresh output
folders:

```powershell
python scripts\run_dominant_trait_pilot.py --dataset datasets\conspired-context1000-pilot\train.jsonl --model hf/Qwen/Qwen2.5-3B-Instruct --model-revision aa8e72537993ba99e69dfaafa59ed015b17504d1 --dtype auto --seed 42 --items-per-class 10 --max-samples 1 --max-tasks 1 --run-environment "RTX 5060 Laptop GPU (8 GB); dtype=auto (BF16); quantization=none; greedy; max-samples=1; max-tasks=1" --results-dir results\dominant-trait-context-pilot-bf16 --log-dir logs\dominant-trait-context-pilot-bf16
```

This makes 120 generations across the two arms. Save the sampled IDs, labels,
document IDs, per-arm outputs, logs, and report. The runner asks for the
upstream-defined most salient/dominant trait in the annotated snippet; context
is used only to interpret the snippet.

## Red-team checks

- On a balanced 60-item set, a constant-majority predictor receives 10/60
  accuracy; include it so a simple policy cannot look like useful performance.
- The source one-hot dominant labels are reused, not independently
  re-annotated. This pilot cannot determine whether a model error is actually
  an upstream label error.
- The pilot comes from training data and may influence prompt revisions. Do
  not use it as confirmatory evidence; keep the official test split untouched.
- Do not use the unmapped `ConspiracyTheoryorMainstream` codes as binary gold
  labels or include them in the prompt.

## Deviations

The first inference run completed all 120 generations, but the runner's parser
required a JSON-quoted label while the model returned bare canonical trait IDs.
The original `report.json` therefore marked every output invalid and must not be
used as a performance readout. The prompt/parser were corrected to accept one
exact canonical ID (or the same ID as a JSON string), and the saved raw
completions were rescored offline; no additional inference was run. The
rescored report is `results/dominant-trait-context-pilot-bf16/report_reparsed.json`.
All 120 completions parse as valid labels. Exact accuracy is 15/60 for
snippet-only and 12/60 for `context1000`; this remains diagnostic only and does
not support a performance or generalization claim.
