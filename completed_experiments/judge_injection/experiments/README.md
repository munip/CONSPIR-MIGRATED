# Prompt-injection detector experiments

Experiments 1–4 are local detector studies. They make no judge calls and do not change
answers, grading prompts, benchmark labels or production defaults. Experiment 5 runs the
judges themselves against injected answers.

| Experiment | Question | Status | Evidence |
|---|---|---|---|
| [1: PromptShield profiles](001-promptshield-profiles/README.md) | Does adding grading rules or ProtectAI v2 improve the pinned baseline? | Completed on the original frozen 140 cases | Existing report and 560 saved scans |
| [2: Prompt Guard 2 86M](002-prompt-guard-2-86m/README.md) | Does a newer classifier improve detection on the same cases? | Completed; 1,260 scans, no scan failures | Frozen protocol, three preserved attempts and result report |
| [3: Dataset expansion](003-dataset-expansion/README.md) | Can we build claim-separated data for English grading-manipulation attempts? | Candidate; human review required before acceptance | Versioned 2,940-case package, blind review records and acceptance gates |
| [4: Local Kev-4B and Laya](004-kev-laya/README.md) | How do the shipped models classify Experiment 3 train items out of the box? | Completed on provisional train labels; in-sample agreement only | Frozen protocol, two model attempts, per-document results and comparison report |
| [5: Judge injection resistance](005-injection-resistance/EXPERIMENT.md) | Do instructions planted in an answer make a judge output the attacker's label? | Completed 2026-10-22 on 70 clean / quoted / attack triples derived from v1; six judges, 231/231 rows each | [RESULTS.md](005-injection-resistance/RESULTS.md), raw results, [interactive explorer](https://spacesheep.dev/@likelytobelaura/judge-injection-resistance) |

Experiments 1 and 2 use `release-v1.0.0.jsonl` and the cumulative assistant-answer
projection in `load_dataset.py`. This release is already exposed to rule design
and calibration. Results show coverage on these cases, not generality.

Experiment 3 prepares new train, validation and test partitions, plus the
unchanged historical fixture as regression data. Test remains a candidate
until required human review passes. Training and publication are deferred.

Experiment 5 derives clean / quoted / attack versions of every v1 pair and asks
Qwen3-4B, pplx-decider, Clef, Kev-4B, Laya and a pplx→Qwen cascade to grade them. Qwen
gave the attacker's label on 19 of 68 attacked pairs (2 of 68 quoted); the other
interpretable judges on at most 2. The fixture is synthetic and exposed, so this is a
regression check, not evidence of general resistance.

Experiment 4 compares local Kev-4B and Laya on the Experiment 3 train split.
It measures agreement with provisional labels; it does not establish attack
success or performance on unseen data. Laya uses a maximum across text windows,
so its document scores are not calibrated probabilities.

## File ownership

| Location | Owner and purpose |
|---|---|
| `../release-v1.0.0.jsonl`, `../manifest-v1.0.0.json`, version 1 recipe/answer/template files | Shared frozen fixture for experiments 1 and 2 and later regression |
| `../load_dataset.py` | Shared case loader and assistant-text projection |
| `001-promptshield-profiles/`, `../reports/promptshield-team-report-v1/` | Experiment 1 record, saved scans and decision report |
| `002-prompt-guard-2-86m/`, `../run_prompt_guard.py`, `../prompt_guard.py`, `../compare_detectors.py` | Experiment 2 protocol, runner, three preserved attempts and comparison code |
| `003-dataset-expansion/` | Experiment 3 authoring, reviews, candidate release, splits and exports |
| `004-kev-laya/` | Experiment 4 protocol, pinned environments, attempts, predictions and reports |
| `005-injection-resistance/` | Experiment 5 runner, derived three-version cases, pinned environment, Modal kit, raw results and RESULTS.md |
| `../run_quality.py`, `../reports/quality-v1.0.0-20261004T163257Z/` | Earlier judge-quality study and its historical report |
| `../reports/promptshield-coverage-v1/` | Historical detector-coverage archive; excluded from numbered comparisons |

Historical paths stay in place because reports and manifests refer to them.
New experiment-specific files belong inside their numbered directory.
