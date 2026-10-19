---
language:
- en
task_categories:
- text-classification
tags:
- synthetic
- prompt-injection
- judge-manipulation
- conspiracybench
pretty_name: ConspiracyBench judge-manipulation candidate 2.0.0
configs:
- config_name: classifier
  data_files:
  - split: train
    path: exports/train.jsonl
  - split: validation
    path: exports/validation.jsonl
  - split: test
    path: exports/test.jsonl
  - split: regression
    path: exports/regression.jsonl
---

# Judge-manipulation candidate 2.0.0

This dataset covers English attempts to manipulate grading in ConspiracyBench assistant answers. It is an AI-authored candidate pending independent human review. It does not establish detection of general reward hacking or multilingual attacks. Do not train or publish a final dataset from this candidate before its review gates pass.

## Composition and sources

The package contains 2,940 cases: 1,960 train, 280 validation, 560 test, and 140 historical regression cases. Every partition balances malicious and benign rows. The 2,800 new cases form 1,400 substantive-answer pairs. Seven attack families each contribute 200 attacks: 140 train, 20 validation and 40 test. Half the new benign rows are hard negatives. Injection and factual labels are independent. A factual error can be benign, and correct-verdict manipulation is malicious.

Accepted source: [Station-house/ConspiracyBench](https://huggingface.co/spaces/Station-house/ConspiracyBench/blob/95827747f30d4f353c5203707b6a2ad996acc34c/questions/corpus.jsonl), frozen at `95827747f30d4f353c5203707b6a2ad996acc34c`. The corpus contains 80 items assigned to 61 claim groups, including framing variants and conservative aliases. All historical fixture claims belong to train. Among the other 41 groups, 29 belong to train, 4 to validation and 8 to test. The regression partition intentionally shares historical claims with train. It is already exposed to previous rule design and calibration.

The [source lock](source/source-lock.json) records the revision, byte hash, attribution and rights status. Source URLs and contributors remain in full case metadata. The [manifest](manifest.json) reports the claims and attack-pattern groups actually represented, content hashes, generation instructions, identity limits, and review status. The [acceptance report](acceptance.json) records checks without confusing structural validity with completed human review.

## Classifier interface

Exports contain exactly three fields: `case_id`, `text`, `label`. Binary label 1 means malicious; 0 means benign. `text` uses the existing `project_scan_input` function: chronological assistant answers joined with two newline characters. Questions, user messages, trusted grading rules, factual labels and review rationales are excluded.

Load four named splits locally:

```python
from datasets import load_dataset

files = {split: f"exports/{split}.jsonl" for split in ("train", "validation", "test", "regression")}
dataset = load_dataset("json", data_files=files)
```

The same JSONL files and named splits support later Hub packaging. See [Hugging Face loading documentation](https://huggingface.co/docs/datasets/loading). No upload occurs in this experiment.

## Authoring and review

Separate Codex author agents wrote source-specific answer seeds and partition-specific attack tactics. A deterministic program assembled pairs and preserved attack spans. The separate reviewer receives full assistant text and trusted source items, with proposed labels hidden. AI decisions, disagreements and method descriptions are retained in `batches/`. Exact serving model revisions are unavailable; provenance records the exposed GPT-6 family and this limit. No independent human authorship is claimed.

The required human queue includes all test rows, all disputes, all source summaries and answer seeds, and a stratified 10% sample of train/validation pairs. Required decisions remain pending until a named human records them with content hashes and independent labels. AI agreement cannot satisfy this gate. Sampled systematic errors require review and repair of the affected batch. [LABELING.md](LABELING.md) describes the policy and record format.

Test remains unfrozen until the strict acceptance check passes. After freezing, training, calibration, threshold selection and error-driven augmentation must use train and validation. New versions must preserve the old test release.

## Limits

Rows share claims, answer cores, and tactics within each partition. Count distinct claims and patterns alongside row totals. Disjoint inventory IDs do not prove semantic independence. Cross-split word 5-gram comparisons surface near duplicates for review; they cannot replace semantic review. Conservative alias groups reduce obvious framing leakage, but the four validation and eight test claims are narrow coverage.

Core length profiles allocate 20% short, 50% typical and 30% long. Hard-negative contexts and attack insertions change final lengths; exact final word/character counts are retained. Long factual cores initially use `hold`, so length and factual class are confounded. Whitespace word offsets beyond 512 provide long-context specimens but do not certify offsets for every future tokenizer. Check the chosen tokenizer later.

A document label does not label every window. Do not label a clean window malicious merely because its parent answer contains an attack elsewhere. Future training must define window labels from visible attack spans and sufficient context. Exclude context-dependent unresolved examples.

This dataset labels attempts. Measuring successful reward hacking requires a separate judge experiment. A balanced synthetic dataset does not estimate deployment precision, attack prevalence or false-positive costs. Forty attacks per family in test support initial comparisons with substantial uncertainty. Naturally generated adversarial answers remain an important later test.

## Rights and publication

The frozen upstream card declares MIT. Source attribution is preserved. Permissive publication is the intended direction, but source contribution rights and generation terms require human confirmation before publication. A model license does not grant rights to this dataset. The card intentionally makes no finalized license claim. Model selection, training, integration and publication are deferred.
