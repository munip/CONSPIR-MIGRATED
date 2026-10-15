#!/usr/bin/env python3
"""Move submissions from questions/pending/*.jsonl into questions/corpus.jsonl,
replacing each cb-pending placeholder with the next free cb-XXXX id. A
maintainer runs this on main after merging one or more submission PRs, never
on a PR branch (two branches would both claim the same next id):

    python scripts/assign_ids.py

Pending files are processed in submission order: the timestamp of the git
commit that first added each file, then file name as a tie-breaker. Appends
to corpus.jsonl, deletes the processed pending files. Run
validate_questions.py afterward.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "questions" / "corpus.jsonl"
PENDING_DIR = ROOT / "questions" / "pending"
ID_PATTERN = re.compile(r"^cb-(\d{4,})$")


def first_commit_time(path: Path) -> int:
    """Unix timestamp of the commit that first added `path`, or a large
    sentinel if the file is not in git history yet (uncommitted), so
    committed submissions always come before uncommitted ones."""
    try:
        out = subprocess.run(
            ["git", "log", "--diff-filter=A", "--follow", "--format=%at", "--", str(path)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
        ).stdout.split()
        return int(out[-1]) if out else 2**62
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 2**62


def main() -> int:
    # Load existing corpus (if any)
    items = []
    if CORPUS.exists():
        # split("\n"), not splitlines(): see validate_questions.py.
        lines = [line.strip() for line in CORPUS.read_text(encoding="utf-8").split("\n") if line.strip()]
        items = [json.loads(line) for line in lines]

    used = set()
    for item in items:
        match = ID_PATTERN.match(item["id"])
        if not match:
            print(f"corpus.jsonl: unexpected id {item['id']!r}. Fix it before assigning.")
            return 1
        used.add(int(match.group(1)))

    # Discover pending files in submission order (git add time, then name)
    pending_files = []
    if PENDING_DIR.exists():
        pending_files = sorted(
            PENDING_DIR.glob("*.jsonl"),
            key=lambda p: (first_commit_time(p), p.name),
        )

    if not pending_files:
        print("No pending files found in questions/pending/. Nothing to do.")
        return 0

    next_id = max(used, default=0) + 1
    assigned = 0
    new_items = []
    for p in pending_files:
        for lineno, line in enumerate(p.read_text(encoding="utf-8").split("\n"), start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("id") != "cb-pending":
                print(f"pending/{p.name}:{lineno}: id must be 'cb-pending', got {item.get('id')!r}. Nothing written.")
                return 1
            while next_id in used:
                next_id += 1
            item["id"] = f"cb-{next_id:04d}"
            used.add(next_id)
            assigned += 1
            new_items.append(item)
            print(f"Assigned {item['id']} from pending/{p.name}: {item['question'][:60]!r}")

    if assigned == 0:
        print("No cb-pending entries found in pending files. Nothing to do.")
        return 0

    # Append to (or create) corpus.jsonl
    all_items = items + new_items
    # newline="\n": text mode would write CRLF on Windows, and the corpus is committed with LF.
    CORPUS.write_text("\n".join(json.dumps(item) for item in all_items) + "\n", encoding="utf-8", newline="\n")

    # Remove processed pending files
    for p in pending_files:
        p.unlink()

    print(f"\n{assigned} id(s) assigned and appended to {CORPUS.name} from {len(pending_files)} pending file(s). Now run scripts/validate_questions.py.")
    return 0


if __name__ == "__main__":
    # Questions can be in any script; a console that cannot show it must not crash the run.
    sys.stdout.reconfigure(errors="backslashreplace")
    raise SystemExit(main())
