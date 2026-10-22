# Run Experiment 005 on the remote GPU host

This procedure runs Qwen, pplx, and Clef on one remote Linux NVIDIA GPU host.
All three use the same pinned Python/PyTorch environment with BF16 weights.
It then derives the cascade and checks that the saved results are complete.
Run commands in order. Use Bash. Commands assume a recent Linux distribution
with `curl`, `tar`, `sha256sum`, `git`, and `tmux` installed.

Start in the repository root on the remote machine. The files from the
experiment PR must already be present, including the shared files under
`completed_experiments/` and `inspect_eval/`. Keep this checkout unchanged
through execution and resume. Use a fresh output directory for this BF16
protocol. Results from the earlier quantized Qwen runtime cannot be resumed
or combined with this run.

For agents: preserve the checkpoint revisions, fixture, prompts, context limits,
and threshold. Do not replace models or change environments to bypass errors.
Stop on a failed setup check. Save partial results if a model run fails.
Do not upload results or send messages without the coordinator's authorization.

## 1. Open a persistent session and set paths

Use a persistent terminal so an SSH disconnect does not stop the run:

```bash
tmux new -s experiment005
```

Set these variables inside that session, from the repository run root:

```bash
set -euo pipefail
export EXP_DIR="$PWD/completed_experiments/judge_injection/experiments/005-injection-resistance"
export ENV_DIR="$EXP_DIR/env"
export RUNNER="$EXP_DIR/experiment.py"
export OUT="$EXP_DIR/results"
export CUDA_VISIBLE_DEVICES=0
```

Use a different GPU index if the operator assigned another device. Set it before
starting Python. The runner requires exactly one visible CUDA device.
Use the same checkout and `OUT` for every command. Do not run two
runner processes against the same output directory.

To detach, press `Ctrl-b`, then `d`. Reconnect with `tmux attach -t experiment005`.
If a new shell is needed, return to the run root and set these variables again.

## 2. Check the host

The target is one 80 GB class NVIDIA GPU with bfloat16 support. H100 80 GB is
the reference target. Plan for 128 GB host RAM and 200 GB free disk space for
snapshots, environments, and temporary downloads. Keep the GPU free of other
model workloads during this experiment.

```bash
uname -s
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
free -h
df -h .
```

The OS must be Linux. The NVIDIA driver must be version 580 or later for the
locked CUDA 13 PyTorch build. The runner checks for an 80 GB class device
and at least 65 GB free VRAM. Stop if the assigned host does not meet the target.
Resolve the host allocation with the operator before proceeding.

## 3. Install the pinned Python environment

If `uv` is absent, use its [official installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Create the experiment environment from its saved lock. Do not run `uv lock` or
update packages on the remote host.

```bash
uv python install 3.12
uv sync --project "$ENV_DIR" --frozen
```

Use this environment for every command below. It contains the five base judges'
runtime dependencies and the production label parser. Each judge runs in its
own Python process and releases its model when that process ends. The runner
sets Python and PyTorch seed 0 for each judge. It keeps the frozen case order.

Check CUDA and bfloat16 through the locked PyTorch build:

```bash
uv run --project "$ENV_DIR" --frozen python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() == 1, 'Expose exactly one CUDA GPU'
assert torch.cuda.is_bf16_supported(), 'bfloat16 is unavailable'
free, total = torch.cuda.mem_get_info(0)
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0), 'free bytes:', free, 'total bytes:', total)
assert total >= 79_000_000_000, 'An 80 GB class GPU is required'
assert free >= 65_000_000_000, 'At least 65 GB free VRAM is required'
PY
```

## 4. Check model access and download pinned snapshots

Sign in with a Hugging Face account that can read all three model repositories.
Use the interactive token prompt. Do not put a token in commands or result files.
Humans run the two authentication commands below. Agents must ask the operator
to sign in when needed; never request or handle the operator's token.

```bash
uv run --project "$ENV_DIR" --frozen hf auth login
uv run --project "$ENV_DIR" --frozen hf auth whoami
uv run --project "$ENV_DIR" --frozen hf download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554
uv run --project "$ENV_DIR" --frozen hf download perplexity-ai/pplx-decider-v1.1-27b \
  --revision 14c9d90137962d300f3de95633f7b6535e80c5ce
uv run --project "$ENV_DIR" --frozen hf download Cloudflare/clef \
  --revision 2f3de3dd85f379784083b0814d997ab627200f0c
```

A 401, 403, or repository access error blocks the run. Ask the coordinator or
repository owner for access. Do not substitute another checkpoint. Downloads
can take substantial time. They do not run model inference.

## 5. Check the fixture, code, and Qwen imports

These commands make no model calls:

```bash
uv run --project "$ENV_DIR" --frozen python -m unittest discover \
  -s "$EXP_DIR" -p 'test_*.py'
uv run --project "$ENV_DIR" --frozen python - <<'PY'
import os
import sys
sys.path.insert(0, os.environ['EXP_DIR'])
import experiment
from inspect_eval.scorer.cb_scorer import parse_label
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
print('Transformers and production parser imports: OK; no model calls')
PY
for judge in qwen pplx clef cascade; do
  uv run --project "$ENV_DIR" --frozen python "$RUNNER" \
    --inspect-only --judge "$judge" --output-dir "$OUT"
done
```

All tests and inspections must pass. Each inspection must show `model_calls: 0`,
`case_rows: 231` (210 three-arm rows plus 21 repeats), `repeat_rows: 21`, this
source fixture hash:

```text
ffee101cd0e4de22b3386940cb0c1559267c2e7aa93c7f927a80c953a46e9b9d
```

and this derived-case hash:

```text
428827302fbcb58346767a0dd4426365360272b11ef936427d068bc8a27df23f
```

Inspection writes a small `summary.json` with completion status and row counts.
Its initial `incomplete` status is expected. Record the checked-out commit for
the results handoff:

```bash
git rev-parse HEAD > "$OUT/source-commit.txt"
```

Do not overwrite this file on resume. Inspection does not prove that a model
fits GPU memory.

## 6. Run Qwen smoke, then the remaining rows

```bash
uv run --project "$ENV_DIR" --frozen python "$RUNNER" \
  --run --judge qwen --limit 3 --output-dir "$OUT"
```

The three rows must have terminal status: `success` or `invalid_output`.
An invalid output is a measured model outcome. An `inference_error` blocks
completion. Check `qwen.manifest.json` and the row error messages if the command
fails. Follow step 10 to resume after resolving the cause.

Continue the same judge in the same directory:

```bash
uv run --project "$ENV_DIR" --frozen python "$RUNNER" \
  --resume --judge qwen --output-dir "$OUT"
nvidia-smi --query-gpu=index,name,memory.free --format=csv
```

Qwen must finish with `status: complete`, `completed_rows: 231`, and
`inference_errors: 0` in its manifest. The Python process releases the model
when the command ends. Wait for GPU memory release before step 7.

## 7. Run pplx, then Clef

Run one command at a time. Each full run resumes its own smoke results.

```bash
uv run --project "$ENV_DIR" --frozen python "$RUNNER" \
  --run --judge pplx --limit 3 --output-dir "$OUT"
uv run --project "$ENV_DIR" --frozen python "$RUNNER" \
  --resume --judge pplx --output-dir "$OUT"
```

Check `pplx.manifest.json`. Require 231 rows and zero inference errors before
continuing. The Python process releases the model when the command ends.

```bash
nvidia-smi --query-gpu=index,name,memory.free --format=csv
uv run --project "$ENV_DIR" --frozen python "$RUNNER" \
  --run --judge clef --limit 3 --output-dir "$OUT"
uv run --project "$ENV_DIR" --frozen python "$RUNNER" \
  --resume --judge clef --output-dir "$OUT"
```

Check `clef.manifest.json`. Require 231 rows and zero inference errors.
`invalid_output` rows remain in the study and do not need retries.

## 8. Derive the cascade and check completion

Do this only after all three base judges finish. The following commands use
saved results and make no inference calls:

```bash
uv run --project "$ENV_DIR" --frozen python "$RUNNER" \
  --run --judge cascade --output-dir "$OUT"
uv run --project "$ENV_DIR" --frozen python "$RUNNER" \
  --inspect-only --judge cascade --output-dir "$OUT"
```

If the cascade already exists and its source files remain unchanged, use
`--resume --judge cascade`. A source change invalidates a saved cascade. Keep
that output directory intact; ask the coordinator before rebuilding changed
sources in a separate directory.

Check the validated completion summary:

```bash
uv run --project "$ENV_DIR" --frozen python - <<'PY'
import json
import os
from pathlib import Path
summary = json.loads((Path(os.environ['OUT']) / 'summary.json').read_text())
assert summary['status'] == 'complete', summary
for judge, counts in summary['models'].items():
    print(judge, counts)
PY
```

Complete status requires all six architectures to have 231 unique rows, with
no missing rows or inference errors. The runner checks manifests, request
hashes, pinned model revisions, and saved cascade derivation before it writes the summary.
`invalid_output` and cascade `upstream_unusable` remain measured outcomes.
Complete status describes execution coverage, not resistance to attacks.

## 9. Return the results

Return the saved results, manifests, completion summary, and source commit.
No Markdown report or metric tables are generated during this run.

```bash
(
  cd "$OUT"
  files=(qwen.results.jsonl qwen.manifest.json
         pplx.results.jsonl pplx.manifest.json clef.results.jsonl clef.manifest.json
         cascade.results.jsonl cascade.manifest.json summary.json source-commit.txt)
  sha256sum "${files[@]}" > SHA256SUMS
  tar -czf "$EXP_DIR/experiment-005-results.tar.gz" "${files[@]}" SHA256SUMS
)
sha256sum "$EXP_DIR/experiment-005-results.tar.gz"
```

Return the archive and its checksum through the agreed channel. Include the
experiment PR URL in the handoff message. Exclude tokens, weights, caches,
`.venv`, terminal logs, and `__pycache__`. The recipient must check the archive
checksum, extract it, and run `sha256sum -c SHA256SUMS` before analysis.

## 10. Resume or diagnose a failed run

A base command returns exit code 1 when it saves inference errors. With
`set -e`, a failure can close the shell. Reattach to tmux or open a new shell,
return to the run root, and set the variables from step 1 again.

Check that the checkout still matches the recorded source commit:

```bash
test "$(git rev-parse HEAD)" = "$(cat "$OUT/source-commit.txt")"
git diff --exit-code HEAD
```

Read the selected judge's manifest and result rows. Fix the environment, model
access, or GPU allocation before retrying. Preserve all pinned revisions and
the environment lock.

```bash
uv run --project "$ENV_DIR" --frozen python "$RUNNER" \
  --resume --judge qwen --output-dir "$OUT"
```

For another judge, use the same environment and its name:

```bash
uv run --project "$ENV_DIR" --frozen python "$RUNNER" \
  --resume --judge pplx --output-dir "$OUT"
```

Use `--judge clef` for Clef. Omit `--limit` when completing a run. Resume retries
missing rows and inference errors. It preserves successful and invalid outputs.
A stopped process can leave `status: running` in the manifest; resume checks and
uses the saved JSONL rows.

Never delete a manifest, edit a result row, truncate a prompt, or change a lock
to bypass a mismatch. If errors persist, preserve all partial files. Return the
partial files with their checksums and state that the experiment is incomplete.
Do not present partial results as a completed comparison.
