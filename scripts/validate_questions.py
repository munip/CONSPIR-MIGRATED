#!/usr/bin/env python3
"""Validate questions/**/*.jsonl against schema.json. Run before opening a PR:

    python scripts/validate_questions.py

Covers both questions/corpus.jsonl and every submission in questions/pending/.
"""
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    sys.exit("Missing dependency: pip install jsonschema")

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "schema.json").read_text())
QUESTIONS_DIR = ROOT / "questions"
PENDING_DIR = QUESTIONS_DIR / "pending"
FORMAT_CHECKER = jsonschema.FormatChecker()
BAND_FOR_TOTAL = {0: "easy", 1: "easy", 2: "medium", 3: "medium", 4: "hard"}


def main() -> int:
    # Include recursive files so we validate corpus.jsonl and pending/*.jsonl
    files = sorted(QUESTIONS_DIR.rglob("*.jsonl"))
    files = [f for f in files if f.name != "template.jsonl"]
    if not files:
        print("No question files found in questions/ (besides template.jsonl).")
        return 1

    seen_ids = set()
    total = 0
    errors = 0

    for path in files:
        label = path.relative_to(QUESTIONS_DIR).as_posix()
        in_pending = PENDING_DIR in path.parents

        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{label}:{lineno}: invalid JSON ({e})")
                errors += 1
                continue

            try:
                jsonschema.validate(item, SCHEMA, format_checker=FORMAT_CHECKER)
            except jsonschema.ValidationError as e:
                print(f"{label}:{lineno}: {e.message}")
                errors += 1
                continue

            total += 1

            # Submissions live in pending/ with the cb-pending placeholder; the
            # maintainer assigns real ids when moving them into corpus.jsonl
            # (see scripts/assign_ids.py). Multiple items may carry cb-pending.
            if in_pending and item["id"] != "cb-pending":
                print(f"{label}:{lineno}: files in pending/ must use id 'cb-pending', got {item['id']!r}")
                errors += 1
            elif not in_pending and item["id"] == "cb-pending":
                print(f"{label}:{lineno}: 'cb-pending' outside pending/; run scripts/assign_ids.py")
                errors += 1
            elif item["id"] != "cb-pending":
                if item["id"] in seen_ids:
                    print(f"{label}:{lineno}: duplicate id {item['id']!r}")
                    errors += 1
                seen_ids.add(item["id"])

            difficulty = item.get("difficulty")
            if isinstance(difficulty, dict):
                score = difficulty["evidence_scarcity"] + difficulty["question_openness"]
                expected_band = BAND_FOR_TOTAL[score]
                if difficulty["band"] != expected_band:
                    print(
                        f"{label}:{lineno}: difficulty.band {difficulty['band']!r} "
                        f"does not match evidence_scarcity + question_openness = {score} "
                        f"(expected {expected_band!r})"
                    )
                    errors += 1

    if errors:
        print(f"\n{errors} error(s).")
        return 1

    print(f"OK — {total} question(s) validated across {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
