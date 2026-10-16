#!/usr/bin/env bash
# From your machine: copy the run's results back. The GH200 is preemptible, so run this every
# so often during the run (e.g. `while sleep 300; do ./pull_results.sh root@<ip>; done`).
set -euo pipefail
host=${1:?usage: pull_results.sh user@host}
repo=$(cd "$(dirname "$0")/../../.." && pwd)
rsync -az "$host":'cb/eval/judge_reliability/results/*_validity_v1*' "$repo/eval/judge_reliability/results/"
rsync -az "$host":judges.log "$repo/eval/judge_reliability/results/decision_judges_v1_run.log" 2>/dev/null || true
ls -1 "$repo"/eval/judge_reliability/results/ | grep _validity_v1
