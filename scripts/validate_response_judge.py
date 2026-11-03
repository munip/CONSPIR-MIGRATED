#!/usr/bin/env python3
"""Compare human reference labels with exported response-robustness judge labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.trait.judge_validation import validate_judgments  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=Path, required=True, help="Human-adjudicated reference JSONL")
    parser.add_argument("--judge", type=Path, required=True, help="Exported judge-output JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Write agreement report JSON")
    args = parser.parse_args(argv)
    try:
        report = validate_judgments(read_jsonl(args.human), read_jsonl(args.judge))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
