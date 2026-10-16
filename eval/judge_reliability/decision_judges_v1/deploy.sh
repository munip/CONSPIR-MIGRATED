#!/usr/bin/env bash
# From your machine: ship the current commit to the GPU host as a git bundle (nothing is pushed
# anywhere), then run setup_host.sh there.
#   eval/judge_reliability/decision_judges_v1/deploy.sh root@<host-ip>
set -euo pipefail
host=${1:?usage: deploy.sh user@host}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
bundle=$(mktemp -d)/cb.bundle

git -C "$repo" diff --quiet HEAD || { echo "commit your changes first: the run records the commit" >&2; exit 1; }
git -C "$repo" bundle create "$bundle" HEAD
scp "$bundle" "$host":cb.bundle
ssh "$host" 'test -d cb || git clone -q cb.bundle cb; cd cb && git fetch -q ../cb.bundle HEAD && git checkout -q --detach FETCH_HEAD && git log -1 --oneline'
ssh "$host" 'bash cb/eval/judge_reliability/decision_judges_v1/setup_host.sh'
