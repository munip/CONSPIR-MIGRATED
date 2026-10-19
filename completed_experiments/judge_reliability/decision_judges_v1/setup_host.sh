#!/usr/bin/env bash
# On the GPU host: check the GPU and driver, build the locked environments, and download all four
# models at their pinned revisions, so no download happens during a timed run. Safe to rerun.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$kit/../../.." && pwd)

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if [ "${driver%%.*}" -lt 580 ]; then
  echo "driver $driver is older than 580; torch 2.14.0's CUDA 13 build needs >= 580" >&2; exit 1
fi
sudo nvidia-smi -pm 1 >/dev/null || echo "warning: could not enable persistence mode"

command -v tmux >/dev/null || sudo apt-get install -y -q tmux
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync --locked --project "$kit/env"
uv sync --locked --project "$kit/env-kev"

uv run --locked --project "$kit/env" python - <<'EOF'
import torch
from huggingface_hub import snapshot_download

assert torch.cuda.is_available(), "torch cannot see the GPU"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
for repo, revision in [
    ("Cloudflare/clef", "2f3de3dd85f379784083b0814d997ab627200f0c"),
    ("jaredpalmer/kev-4b", "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"),
    ("convaiinnovations/laya", "7b928d828b7b0e022f929d9bd2e44165aa270148"),
    ("perplexity-ai/pplx-decider-v1.1-27b", "14c9d90137962d300f3de95633f7b6535e80c5ce"),
]:
    print(repo, "->", snapshot_download(repo, revision=revision))
EOF
# Kev's SDK resolves its base model on first load; one tiny smoke call later caches it too.
echo "setup done in $repo"
