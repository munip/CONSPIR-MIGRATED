#!/usr/bin/env python3
"""Assign real ids to any cb-pending placeholder entries in
questions/examples.jsonl, in file order. A maintainer runs this once,
right before merging a PR (or a batch of PRs) that used the cb-pending
placeholder instead of guessing the next number:

    python scripts/assign_ids.py

Rewrites the file in place. Run validate_questions.py afterward.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "questions" / "examples.jsonl"
ID_PATTERN = re.compile(r"^cb-(\d{4,})$")


def main() -> int:
    lines = [line.strip() for line in EXAMPLES.read_text().splitlines() if line.strip()]
    items = [json.loads(line) for line in lines]

    used = set()
    for item in items:
        match = ID_PATTERN.match(item["id"])
        if match:
            used.add(int(match.group(1)))

    next_id = max(used, default=0) + 1
    assigned = 0
    for item in items:
        if item["id"] != "cb-pending":
            continue
        while next_id in used:
            next_id += 1
        item["id"] = f"cb-{next_id:04d}"
        used.add(next_id)
        assigned += 1
        print(f"Assigned {item['id']}: {item['question'][:60]!r}")

    if assigned == 0:
        print("No cb-pending entries found. Nothing to do.")
        return 0

    EXAMPLES.write_text("\n".join(json.dumps(item) for item in items) + "\n")
    print(f"\n{assigned} id(s) assigned. Now run scripts/validate_questions.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
