#!/usr/bin/env python3
"""Export per-turn response-robustness judgments from an Inspect .eval log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_rows(log, scorer_name: str = "response_robustness_scorer") -> list[dict]:
    rows = []
    for sample in log.samples or []:
        score = (sample.scores or {}).get(scorer_name)
        if score is None:
            continue
        try:
            judgments = json.loads(str(score.value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse scorer value for sample {sample.id}") from exc
        if not isinstance(judgments, list):
            raise ValueError(f"Scorer value for sample {sample.id} is not a judgment list")
        for turn_index, judgment in enumerate(judgments):
            if not isinstance(judgment, dict):
                raise ValueError(f"Sample {sample.id} turn {turn_index} is not a judgment object")
            rows.append({
                "sample_id": judgment.get("sample_id") or str(sample.id),
                "epoch": judgment.get("epoch", sample.epoch or 1),
                "turn_index": judgment.get("turn_index", turn_index),
                **{key: judgment[key] for key in (
                    "factual_alignment", "conspiracy_endorsement", "abstention", "invalid"
                ) if key in judgment},
            })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Inspect response-robustness .eval log")
    parser.add_argument("--output", type=Path, required=True, help="Output UTF-8 JSONL")
    args = parser.parse_args(argv)
    try:
        from inspect_ai.log import read_eval_log
        rows = export_rows(read_eval_log(str(args.log)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Exported {len(rows)} per-turn judgments to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
