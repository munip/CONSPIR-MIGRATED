# Experiment 4: Local Kev-4B and Laya comparison

This experiment compares the two shipped models on the 1,960 Experiment 3 train rows. The release is an AI-authored candidate pending human review. The results measure agreement with its provisional labels. They do not establish attack success or performance on unseen data.

## Frozen protocol

The complete protocol is in [protocol.json](protocol.json). The runner verifies the export hash, IDs, assistant-text projection, and labels against the full Experiment 3 release before inference. It sends only the exported text and the fixed `noul` question to a local model. It joins train metadata after inference for analysis.

Both models use one shuffled pass with seed `0` and the `yes >= 0.5` decision rule. Kev uses the pinned SDK and checkpoint through its MLX backend with BF16 weights and shipped calibration. It reads the complete document without truncation or prefix reuse. Laya uses its pinned English checkpoint through PyTorch MPS with the SDK's default precision policy and stock `predict_long` settings. Laya's document score is the maximum window yes probability. The report groups Laya results by window count because this maximum is not a calibrated document probability.

Five fixed synthetic inputs warm each model before scoring. They are not included in scored results. Model loading, full token coverage, backend, and precision are checked before the pass. A memory, initialization, or device-fallback failure marks the attempt incomplete and leaves partial files in place.

## Environments and execution

The model environments have separate `pyproject.toml` and `uv.lock` files. Both use Python 3.12. After generating or refreshing the locks with `uv lock --project envs/kev` and `uv lock --project envs/laya`, install and run both models sequentially with:

```sh
./run_sequential.sh
```

This starts Kev first and Laya second in separate processes. The script continues to Laya if Kev's attempt is incomplete. Each process creates a new directory under `attempts/`. After both finish, it rebuilds the comparison from saved results under `reports/`.

Run one model with its own locked environment:

```sh
uv run --locked --project envs/kev python runner.py --model kev
uv run --locked --project envs/laya python runner.py --model laya
```

Rebuild reports without inference:

```sh
python3 report.py --latest
```

## Outputs

Each attempt stores `manifest.json`, `results.jsonl`, `preflight.json`, and `warmup.json`. Results include explicit inference failures and omit labels and metadata. The report builder joins results to the verified release and writes `reports/experiment-004-summary.json` and `reports/experiment-004-comparison.md`.

The report includes successful-result accuracy and coverage, correct predictions divided by all 1,960 rows, precision, recall, F1, false-positive rate, confusion matrix, recall by attack family, hard-negative false positives, latency percentiles, runtime, memory, and artifact size. The summary JSON stores the exact error IDs. The report includes per-window scores and results grouped by Laya window count.

Model-free verification:

```sh
cd eval/judge_injection/experiments/004-kev-laya
python3 test_experiment.py
```

No model call is needed for the tests or report rebuild.
