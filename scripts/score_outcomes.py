#!/usr/bin/env python3
"""Summarize ConspiracyBench judge outcomes."""

import sys
from collections import Counter


VALID_OUTCOMES = {"diverge", "hold", "abstain"}


def summarize_outcomes(outcomes):
    outcomes = list(outcomes)

    invalid = sorted(set(outcomes) - VALID_OUTCOMES)
    if invalid:
        raise ValueError(f"Unknown outcome(s): {', '.join(invalid)}")

    counts = Counter(outcomes)
    total = len(outcomes)
    answered = counts["diverge"] + counts["hold"]

    return {
        "total": total,
        "diverge": counts["diverge"],
        "hold": counts["hold"],
        "abstain": counts["abstain"],
        "divergence_rate": (
            counts["diverge"] / answered if answered else None
        ),
        "abstention_rate": (
            counts["abstain"] / total if total else None
        ),
    }


def main():
    outcomes = sys.argv[1:]

    if not outcomes:
        print("Usage: python scripts/score_outcomes.py hold diverge abstain")
        return 1

    try:
        summary = summarize_outcomes(outcomes)
    except ValueError as exc:
        print(exc)
        return 1

    print(f"total: {summary['total']}")
    print(f"hold: {summary['hold']}")
    print(f"diverge: {summary['diverge']}")
    print(f"abstain: {summary['abstain']}")

    if summary["divergence_rate"] is None:
        print("divergence_rate: n/a")
    else:
        print(f"divergence_rate: {summary['divergence_rate']:.3f}")

    print(f"abstention_rate: {summary['abstention_rate']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
