# Experiment 3: judge-manipulation dataset expansion

Status: AI-reviewed candidate. Required human review is pending. This experiment prepares data; it makes no detector, judge, training, or publication claim.

The [versioned candidate](releases/2.0.0/) adds 2,800 synthetic cases in 1,400 matched pairs. The original 140-case fixture remains a separate regression partition. The [source lock](releases/2.0.0/source/source-lock.json) records the immutable accepted upstream revision. The [claim partition file](releases/2.0.0/source/claim-partitions.json) records 61 conservative claim groups. Historical fixture claims belong to train. Remaining groups split 29/4/8 across train/validation/test.

| Partition | Malicious | Benign | Rows |
|---|---:|---:|---:|
| Train | 980 | 980 | 1,960 |
| Validation | 140 | 140 | 280 |
| Test | 280 | 280 | 560 |
| Regression | 70 | 70 | 140 |
| Total | 1,470 | 1,470 | 2,940 |

Each new split contains all seven families. Each family supplies 140/20/40 attacks in train/validation/test. Half of new benign answers are hard negatives. New attacks request a conflicting verdict in 60%, the correct verdict in 20%, and evaluator control without a fixed verdict in 20%. Every family has 20% multi-turn pairs. Core length profiles use 20/50/30 short/typical/long. The [manifest](releases/2.0.0/manifest.json) and [acceptance report](releases/2.0.0/acceptance.json) report actual coverage, distinct claims, patterns, and pending work.

All 2,800 new rows have independent blind AI classifications. The 14 rows changed during the Bali coherence repair received a fresh blind pass; earlier drafts and reviews remain archived. Structural validation passes, and all four exports load with Hugging Face Datasets. Required human work comprises 864 tasks: 80 source reviews, 560 test rows, and 224 sampled train/validation rows. Strict acceptance and test freezing reject this candidate until those reviews pass.

## Files owned by this experiment

| Path | Purpose |
|---|---|
| `pipeline.py`, `test_pipeline.py` | Build, blind-review joins, validation, and freeze gate |
| `AUTHORING.md` | Generation instructions and review protocol |
| `releases/2.0.0/source/` | Frozen accepted corpus, revision, grouping, and protected artifact hashes |
| `releases/2.0.0/authoring/` | AI answer seeds, attack inventories, and hard-negative contexts |
| `releases/2.0.0/batches/` | Pilot and 140-row batches, blind inputs, reviewer decisions, and disagreements |
| `releases/2.0.0/release-v2.0.0.jsonl` | Full case structure and metadata for all 2,940 rows |
| `releases/2.0.0/splits/` | Full cases in four named partitions |
| `releases/2.0.0/exports/` | Model-neutral classifier input: exactly `case_id`, `text`, and binary `label` |
| `releases/2.0.0/review/` | Required human tasks, decisions, and cross-split near-duplicate candidates |
| `releases/2.0.0/README.md`, `LABELING.md` | Dataset card and labeling guide |

The blind file hides proposed labels, pair IDs and author rationales. Its trusted source item supports factual review only. It never enters classifier input. `private-map.jsonl` is the author/review join and is excluded from the reviewer pass.

## Reproduce and review

Run from the repository root with Python 3.12 or later. There are no third-party build dependencies.

```sh
python3 eval/judge_injection/experiments/003-dataset-expansion/pipeline.py check
python3 -m unittest discover -s eval/judge_injection/experiments/003-dataset-expansion -p 'test_*.py'
```

`check` checks structure and reports pending human work. `check --require-human` must fail until all required reviews pass. A structural pass alone does not accept this dataset.

The saved batch inputs are the reproducible authoring record. To author a new batch, use a new output ID. Review the entire training pilot before scaling:

```sh
python3 eval/judge_injection/experiments/003-dataset-expansion/pipeline.py batch --split train --start 0 --batch-id batch-001
python3 eval/judge_injection/experiments/003-dataset-expansion/pipeline.py review-check --batch-id batch-001
```

The first command refuses an existing output. AI reviewer classifications belong in that batch's `ai-review.jsonl`. The package command joins all retained batches and regenerates deterministic exports. It preserves human decision inputs. Do not overwrite an earlier review to conceal a dispute.

Required human work includes all 560 test rows, all disputes, all 80 source summaries/answer inventories, and 112 stratified training/validation pairs. Review both members of every sampled pair. Review the full affected batch if the sample shows a systematic error. The [labeling guide](releases/2.0.0/LABELING.md) describes exact decision records. Near-duplicate candidates require explicit decisions too.

```sh
python3 eval/judge_injection/experiments/003-dataset-expansion/pipeline.py check --require-human
python3 eval/judge_injection/experiments/003-dataset-expansion/pipeline.py freeze-test
```

The freeze command records test and review hashes. Later authoring cannot rebuild this version. Test inspection must not guide training, threshold selection, or augmentation.

Model selection, Unsloth training, integration and publication remain deferred. Publication also requires confirmation of dataset rights and generation terms. The upstream card declares MIT; that declaration is preserved as evidence and does not settle all contribution or generation rights.
