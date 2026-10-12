# Experiment 2: Prompt Guard 2 86M

Status: completed. [Attempt 3 results](attempts/run-003/README.md) contain
1,260 scans on the original 140 cases. [Designed report](attempts/run-003/report.html)
includes all metrics, four charts, exact case errors and the Experiment 1 comparison.

| Profile / cutoff | Attacks caught | Benign FPs | Quoted FPs |
|---|---:|---:|---:|
| Prompt Guard alone / 0.5 | 29/70 | 19/70 | 19/21 |
| Baseline + Prompt Guard / 0.5 | 36/70 | 20/70 | 19/21 |
| Baseline + Prompt Guard / selected 0.997957706451416 | 22/70 | 2/70 | 1/21 |
| Rules + baseline | 70/70 | 2/70 | 1/21 |
| Rules + baseline + Prompt Guard / 0.5 | 70/70 | 20/70 | 19/21 |

Prompt Guard does not beat rules on this fixture. At the two-FP budget it adds
nine detections to baseline, but rules still catches all attacks. The default
model flags quoted/rejected attack text. Selected cutoffs are in-sample and must
be fixed before the later generality test set is evaluated.

Warm CPU p50/p95 across three shuffled passes: Prompt Guard alone 49.5/445.2 ms;
rules + baseline 22.2/144.4 ms; baseline 22.0/144.0 ms. OR combinations above
have derived detection results and no measured end-to-end latency. All repeated
passes agree on detection actions. The host is Apple M5, 24 GiB RAM, macOS 26.4;
CPU inference uses two threads. Power/thermal conditions were not controlled.

Attempts are preserved: [1](attempts/run-001/summary.json) failed with HTTP 403;
[2](attempts/run-002/summary.json) loaded the checkpoint but stopped before scans
because its labels are generic. Meta's pinned inference helper verifies malicious
index 1; the loader was corrected before [3](attempts/run-003/summary.json).
The model revision and frozen protocol stayed unchanged. Attempt 3 calibration
hit a baseline-score validation bug; this was repaired from saved scores without
rerunning inference. Native summary and scans are preserved; [analysis.json](attempts/run-003/analysis.json)
contains the secondary calibration. There were no scan failures or judge calls.

See [protocol.json](protocol.json) for the settings fixed before inference.

The run's [source archive](attempts/run-003/source.zip) includes the inference code and
its dependency files. To repeat the scan, extract it into a separate working copy, run
these commands from the extracted root, and install the recorded ML requirements:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r inspect_eval/requirements-promptshield-ml.txt
OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false python \
  eval/judge_injection/run_prompt_guard.py \
  --output-dir eval/judge_injection/experiments/002-prompt-guard-2-86m/attempts/run-004
```

Each attempt uses a new output directory. Failed initialization and inference
are saved explicitly. Existing attempts are never overwritten.

The study uses only the same frozen 140 cases as Experiment 1. It evaluates
Prompt Guard alone and repeats baseline/rules CPU scans for timing context.
Three shuffled timing passes use fixed seeds. Default cutoff 0.5 is evaluated
first. Baseline + Prompt Guard and rules + Prompt Guard detection results are
computed from saved findings; their end-to-end latency is not measured.

A threshold curve and a cutoff selected within the same two-benign-FP budget are
secondary, in-sample results. Raw default actions and scores remain unchanged.
Record all false-positive and missed case IDs, family recall, quoted-control
errors, load failures and warm scan timings. No judge calls, model fine-tuning,
production default changes or new generality cases are part of this experiment.

Validation: ten model-free scanner/runner tests and thirteen existing study
tests pass. A mock full run produces the planned 1,260 timing records. These
checks verify the runner, not Prompt Guard detection quality. Another 74 adapter,
dependency, scorer and report tests pass. Independent checks reproduce detection
totals, confirm stable actions across passes, validate report links and verify
all attempt and Experiment 1 artifact hashes. Failed attempts retain source
hashes, exact source archives, runtime versions and errors.

Rebuild the report from saved results without inference. The report builder needs NumPy
and Matplotlib:

```sh
python -m pip install numpy matplotlib
python \
  eval/judge_injection/experiments/002-prompt-guard-2-86m/build_report.py \
  --attempt eval/judge_injection/experiments/002-prompt-guard-2-86m/attempts/run-003
```
