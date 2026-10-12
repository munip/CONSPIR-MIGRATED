#!/usr/bin/env bash
# Usage: prep_pr.sh [--run-pr-scripts] <PR number> [path to your ConspiracyBench clone]
#
# Fetches a ConspiracyBench PR into a throwaway git worktree and prints the PR
# discussion and the PR's pending item(s). The checks run main's copies of
# validate_questions.py and check_duplicates.py, against main's schema and corpus
# plus only the items this PR adds, in a second throwaway worktree. Code in the
# PR never runs unless you pass --run-pr-scripts, so read any code the PR changes
# before you do.
#
# It never checks out or modifies your clone's working tree, so your own unsubmitted
# files in questions/pending/ stay safe. It posts nothing to Hugging Face.
#
# Needs: bash, git, curl, python3. Windows users: run it under WSL or Git Bash.
set -euo pipefail

RUN_PR_SCRIPTS=0
if [ "${1:-}" = "--run-pr-scripts" ]; then RUN_PR_SCRIPTS=1; shift; fi

N="${1:?usage: prep_pr.sh [--run-pr-scripts] <PR number> [repo path]}"
REPO="${2:-${CB_REPO:-$(git rev-parse --show-toplevel 2>/dev/null || true)}}"
if [ -z "$REPO" ] || [ ! -f "$REPO/schema.json" ]; then
  echo "Can't find a ConspiracyBench clone. Run from inside one, pass its path, or set CB_REPO." >&2
  exit 1
fi

TMP="${TMPDIR:-/tmp}"
WT="${TMP%/}/cb-pr-$N"        # the PR as submitted, for reading
BASE="${TMP%/}/cb-pr-$N-main" # main + this PR's pending items; main's scripts run here
VENV="${XDG_CACHE_HOME:-$HOME/.cache}/conspiracybench-review/venv"

echo "== Discussion #$N"
curl -s "https://huggingface.co/api/spaces/Station-house/ConspiracyBench/discussions/$N" | python3 -I -c '
import json,sys
d=json.load(sys.stdin)
print("title:",d.get("title"),"| status:",d.get("status"),"| author:",(d.get("author") or {}).get("name"))
for e in d.get("events",[]):
    if e.get("type")=="comment":
        print("\n-- comment by",(e.get("author") or {}).get("name"))
        print(e.get("data",{}).get("latest",{}).get("raw",""))
'

echo; echo "== Fetching refs/pr/$N"
git -C "$REPO" fetch -q origin "+main:refs/remotes/origin/main" "+refs/pr/$N:refs/remotes/origin/pr/$N"
RANGE="origin/main...origin/pr/$N"
for w in "$WT" "$BASE"; do git -C "$REPO" worktree remove --force "$w" 2>/dev/null || true; done
git -C "$REPO" worktree add -q --detach "$WT" "origin/pr/$N"
git -C "$REPO" worktree add -q --detach "$BASE" origin/main
echo "PR worktree:   $WT"
echo "Main worktree: $BASE"
echo; echo "== Files changed vs main"
git -C "$REPO" diff --stat "$RANGE"

# Only the pending items this PR adds or edits, not everything already waiting on main.
PENDING=()
while IFS= read -r f; do [ -n "$f" ] && PENDING+=("$f"); done < <(
  git -C "$REPO" diff --name-only --diff-filter=AM "$RANGE" -- 'questions/pending/*.jsonl')
rm -f "$BASE"/questions/pending/*.jsonl
for f in ${PENDING[@]+"${PENDING[@]}"}; do cp "$WT/$f" "$BASE/$f"; done

CODE=()
while IFS= read -r f; do [ -n "$f" ] && CODE+=("$f"); done < <(
  git -C "$REPO" diff --name-only "$RANGE" -- 'scripts/' '*.py' '*.sh')
if [ ${#CODE[@]} -gt 0 ]; then
  echo
  echo "!! WARNING: this PR changes code that a reviewer might run:"
  printf '!!   %s\n' "${CODE[@]}"
  echo "!! The checks below use main's copies, not these. Read the changes first:"
  echo "!!   git -C \"$REPO\" diff $RANGE -- ${CODE[*]}"
  if [ "$RUN_PR_SCRIPTS" = 0 ]; then
    echo "!! To also run the PR's versions once you've read them, re-run with --run-pr-scripts."
  fi
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo; echo "== Creating review venv at $VENV (first run only; fastembed downloads a ~64MB model)"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" -q install jsonschema fastembed numpy
fi

run_checks() {
  echo; echo "== validate_questions.py ($2)"
  (cd "$1" && "$VENV/bin/python" scripts/validate_questions.py 2>&1 | tail -15) || true
  echo; echo "== check_duplicates.py ($2)"
  (cd "$1" && "$VENV/bin/python" scripts/check_duplicates.py 2>&1) || true
}

if [ ${#PENDING[@]} -eq 0 ]; then
  echo; echo "== No pending items added or edited; skipping validation and duplicate checks."
else
  run_checks "$BASE" "main's copy, on this PR's items"
fi
if [ "$RUN_PR_SCRIPTS" = 1 ]; then
  echo; echo "!! Running the PR's own scripts, as requested with --run-pr-scripts."
  run_checks "$WT" "PR's copy, on the PR as submitted"
fi

echo; echo "== Pending item(s) added or edited by this PR, pretty-printed"
[ ${#PENDING[@]} -gt 0 ] || echo "(none)"
for f in ${PENDING[@]+"${PENDING[@]}"}; do
  [ -s "$WT/$f" ] || continue
  echo "-- $f"
  python3 -I -c 'import json,sys
for line in open(sys.argv[1]):
    line=line.strip()
    if line: print(json.dumps(json.loads(line),indent=2,ensure_ascii=False))' "$WT/$f"
done

echo; echo "When done: git -C \"$REPO\" worktree remove --force \"$WT\" && git -C \"$REPO\" worktree remove --force \"$BASE\""
