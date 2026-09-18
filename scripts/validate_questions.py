#!/usr/bin/env python3
"""Validate questions/*.jsonl against schema.json. Run before opening a PR:

    python scripts/validate_questions.py
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
FORMAT_CHECKER = jsonschema.FormatChecker()


def main() -> int:
    files = sorted(QUESTIONS_DIR.glob("*.jsonl"))
    files = [f for f in files if f.name != "template.jsonl"]
    if not files:
        print("No question files found in questions/ (besides template.jsonl).")
        return 1

    seen_ids = set()
    errors = 0

    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{path.name}:{lineno}: invalid JSON ({e})")
                errors += 1
                continue

            try:
                jsonschema.validate(item, SCHEMA, format_checker=FORMAT_CHECKER)
            except jsonschema.ValidationError as e:
                print(f"{path.name}:{lineno}: {e.message}")
                errors += 1
                continue

            if item["id"] in seen_ids:
                print(f"{path.name}:{lineno}: duplicate id {item['id']!r}")
                errors += 1
            seen_ids.add(item["id"])

    if errors:
        print(f"\n{errors} error(s).")
        return 1

    print(f"OK — {len(seen_ids)} question(s) validated across {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
