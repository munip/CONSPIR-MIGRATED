#!/usr/bin/env python3
"""Results tables from decision-judge grades, in the same shape as ``report.py``'s.

    python inspect_eval/decision_judges/decision_report.py graded/<tag> --out results/<tag>

``graded/<tag>`` holds ``units.jsonl`` and one folder per judge with its ``results.jsonl`` (as
``modal_grade.py`` leaves them). Each judge's verdicts are put back into trajectories, one per
sample per epoch, and summarized by ``report.summarize_rows``: headline rates with Wilson and
clustered intervals, concession and retreat, pass@k / worst-of-k / all-k-agree, invalid turns
by reason and the breakdowns. A turn the judge couldn't grade, or that wasn't graded (``skip``),
is invalid, as in the Inspect scorer.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "eval"))

from cb_budget import INVALID  # noqa: E402
import report  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def labels_by_case(rows: list[dict]) -> dict:
    """case_id -> (label, invalid reason). The last row for a unit wins."""
    return {row["case_id"]: (row["choice"], None) if row.get("status") == "ok" else (INVALID, "judge_error")
            for row in rows}


def rows_from_grades(units: list[dict], labels: dict) -> list[dict]:
    """One row per sample per epoch, as ``report.rows_from_log`` makes them."""
    by_sample = defaultdict(list)
    for unit in units:
        by_sample[(unit["sample_id"], unit["epoch"])].append(unit)
    rows = []
    for (sample_id, epoch), turns in sorted(by_sample.items()):
        turns.sort(key=lambda u: u["turn"])
        graded = [(INVALID, u["skip"]) if u.get("skip") else labels.get(u["case_id"], (INVALID, "not_graded"))
                  for u in turns]
        breakdown = turns[0].get("breakdown") or {}
        rows.append({
            "id": sample_id,
            "epoch": epoch,
            "trajectory": [label for label, _ in graded],
            "invalid_reasons": [reason for _, reason in graded],
            "truncated_turns": [],
            "cluster": turns[0]["item"].get("claim_id") or f"item:{sample_id}",
            **{field: breakdown.get(field) for field in report.BREAKDOWNS},
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("graded", type=Path, help="folder with units.jsonl and one folder per judge")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    units = load_jsonl(args.graded / "units.jsonl")
    judges = [d for d in sorted(args.graded.iterdir()) if (d / "results.jsonl").exists()]
    if not judges:
        sys.exit(f"no <judge>/results.jsonl under {args.graded}")
    args.out.mkdir(parents=True, exist_ok=True)
    # A units file can hold several evals (export with a "variant" per unit); each gets its own table.
    variants = sorted({u.get("variant") for u in units}, key=str)
    for judge in judges:
        labels = labels_by_case(load_jsonl(judge / "results.jsonl"))
        for variant in variants:
            rows = rows_from_grades([u for u in units if u.get("variant") == variant], labels)
            summary = {"graded": str(args.graded), "judge": judge.name, "variant": variant, **report.summarize_rows(rows)}
            path = args.out / (f"{judge.name}_{variant}.json" if variant else f"{judge.name}.json")
            path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
            head, rep = summary["headline"], summary["repeated_sampling"] or {}
            conv = summary.get("repeated_conversations") or {}
            print(f"{judge.name} {variant or ''}: divergence {head['divergence_rate']['rate']}, "
                  f"abstention {head['abstention_rate']['rate']}, concession {head['concession_rate']['rate']}, "
                  f"pass@k {rep.get('pass_at_k', {}).get('rate')}, worst-of-k {rep.get('worst_of_k', {}).get('rate')}, "
                  f"conversation pass@k {conv.get('pass_at_k', {}).get('rate')} -> {path}")


if __name__ == "__main__":
    main()
