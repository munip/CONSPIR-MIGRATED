#!/usr/bin/env bash
# On the GPU host, inside tmux: a 3-call smoke test of every judge, then the full run of every
# judge, in the fixed order of protocol.json, one process each, with the same runner arguments.
# Rerunning resumes: finished calls are skipped, and each new process logs a new session.
#   tmux new -d -s judges 'bash cb/eval/judge_reliability/decision_judges_v1/run_all.sh 2>&1 | tee -a judges.log'
#   STAGE=smoke or STAGE=full runs one stage only.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$kit/../../.." && pwd)
export PATH="$HOME/.local/bin:$PATH"
judges=(clef kev laya pplx)
units=eval/judge_reliability/validity_v1/units.jsonl
stage=${STAGE:-all}
cd "$repo"

env_for() {
  # Kev uses env-kev only if KEV_ENV=env-kev was set after a failed smoke test under env.
  if [ "$1" = kev ] && [ "${KEV_ENV:-env}" = env-kev ]; then echo "$kit/env-kev"; else echo "$kit/env"; fi
}

gpu_idle() {
  local busy
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$busy" -eq 0 ] || { echo "GPU is busy ($busy processes); refusing to time a run on a shared GPU" >&2; exit 1; }
}

run_judge() {  # judge out_dir extra-args...
  local judge=$1 out=$2; shift 2
  mkdir -p "$out"
  gpu_idle
  # One sample per second for the whole process: utilisation, memory, power, clocks, temperature.
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu \
    --format=csv,nounits -l 1 >> "$out/gpu_telemetry.csv" &
  local smi=$!
  local status=0
  uv run --locked --project "$(env_for "$judge")" python eval/judge_reliability/run_decision_validity.py \
    --judge "$judge" --units "$units" --output-dir "$out" --device cuda --warmup 3 --run "$@" || status=$?
  kill "$smi" 2>/dev/null || true
  uv pip freeze --python "$(env_for "$judge")/.venv/bin/python" > "$out/pip_freeze.txt" 2>/dev/null || true
  return $status
}

if [ "$stage" != full ]; then
  for judge in "${judges[@]}"; do
    out=eval/judge_reliability/results/${judge}_validity_v1_smoke
    run_judge "$judge" "$out" --limit 3
    if grep -q '"status": "error"' "$out/results.jsonl"; then
      echo "$judge smoke test has errors:" >&2; grep '"status": "error"' "$out/results.jsonl" >&2; exit 1
    fi
  done
  echo "smoke tests passed"
fi

if [ "$stage" != smoke ]; then
  for judge in "${judges[@]}"; do
    run_judge "$judge" "eval/judge_reliability/results/${judge}_validity_v1"
  done
  uv run --locked --project "$kit/env" python "$kit/speed_report.py" --price-per-hour 1.99 \
    eval/judge_reliability/results/{clef,kev,laya,pplx}_validity_v1
fi
