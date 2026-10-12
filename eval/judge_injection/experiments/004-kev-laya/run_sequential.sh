#!/bin/sh

set +e
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

uv run --locked --project "$HERE/envs/kev" python "$HERE/runner.py" --model kev
KEV_STATUS=$?

uv run --locked --project "$HERE/envs/laya" python "$HERE/runner.py" --model laya
LAYA_STATUS=$?

python3 "$HERE/report.py" --latest
REPORT_STATUS=$?

if [ "$KEV_STATUS" -ne 0 ] || [ "$LAYA_STATUS" -ne 0 ] || [ "$REPORT_STATUS" -ne 0 ]; then
    exit 1
fi
