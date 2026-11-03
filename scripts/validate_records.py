#!/usr/bin/env python3
"""Validate legacy ConspiracyBench questions and versioned CONSPIR JSONL records.

Examples:
    python scripts/validate_records.py questions/corpus.jsonl
    python scripts/validate_records.py datasets/conspired/normalized.jsonl datasets/loco
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    raise SystemExit("Missing dependency: pip install jsonschema")

ROOT = Path(__file__).resolve().parent.parent
LEGACY_SCHEMA = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
VERSIONED_SCHEMA = json.loads(
    (ROOT / "schemas" / "conspir" / "v2_schema.json").read_text(encoding="utf-8")
)
FORMAT_CHECKER = jsonschema.FormatChecker()
VALIDATORS = {
    id(LEGACY_SCHEMA): jsonschema.Draft7Validator(LEGACY_SCHEMA, format_checker=FORMAT_CHECKER),
    id(VERSIONED_SCHEMA): jsonschema.Draft7Validator(VERSIONED_SCHEMA, format_checker=FORMAT_CHECKER),
}


def schema_for(record: dict) -> dict:
    """Keep old question records on their existing schema; route new kinds by version."""
    if "schema_version" in record or "task_type" in record:
        return VERSIONED_SCHEMA
    return LEGACY_SCHEMA


def record_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file() and path.suffix.lower() == ".jsonl":
            files.append(path)
        else:
            raise FileNotFoundError(f"Not a JSONL file or directory: {path}")
    return list(dict.fromkeys(files))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        files = record_files(args.paths)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not files:
        print("No JSONL records found.")
        return 1

    seen_ids = set()
    count = errors = 0
    for path in files:
        with path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"{path}:{line_no}: invalid JSON ({exc})")
                    errors += 1
                    continue
                if not isinstance(record, dict):
                    print(f"{path}:{line_no}: record must be a JSON object")
                    errors += 1
                    continue
                try:
                    schema = schema_for(record)
                    VALIDATORS[id(schema)].validate(record)
                except jsonschema.ValidationError as exc:
                    print(f"{path}:{line_no}: {exc.json_path}: {exc.message}")
                    errors += 1
                    continue
                identifier = record.get("id")
                if identifier in seen_ids:
                    print(f"{path}:{line_no}: duplicate id {identifier!r}")
                    errors += 1
                seen_ids.add(identifier)
                count += 1

    if errors:
        print(f"\n{errors} error(s) across {count} valid record(s).")
        return 1
    print(f"OK - {count} record(s) validated across {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="backslashreplace")
    raise SystemExit(main())
