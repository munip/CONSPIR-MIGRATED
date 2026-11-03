#!/usr/bin/env python3
"""Write a compact per-trait, per-source, and per-domain report from an Inspect log."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from trait.trait_classification import _metric_summary  # noqa: E402


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def report_from_log(log, scorer_name: str = "trait_scorer") -> dict:
    scored = []
    for sample in log.samples or []:
        score = (sample.scores or {}).get(scorer_name)
        if score is not None:
            scored.append(SimpleNamespace(score=score))
    if not scored:
        raise ValueError(f"Inspect log contains no {scorer_name!r} scores")
    summary = _metric_summary(scored)
    return _json_safe({
        "scorer": scorer_name,
        "log_samples": len(log.samples or []),
        "scored_samples": len(scored),
        "overall": {
            "per_trait": summary["per_trait"],
            "macro_f1": summary["macro_f1"],
            "micro_f1": summary["micro_f1"],
            "exact_match": summary["exact_match"],
            "dominant_trait_hit_rate": summary["dominant_trait_hit_rate"],
            "dominant_trait_support": summary["dominant_trait_support"],
            "dominant_trait_reference_support": summary["dominant_trait_reference_support"],
            "valid_output_count": summary["valid_output_count"],
            "total_output_count": summary["total_output_count"],
            "invalid_output_rate": summary["invalid_output_rate"],
        },
        "by_domain": summary["by_domain"],
        "by_domain_reference_support": summary["by_domain_reference"],
        "by_source": summary["by_source"],
        "by_source_reference_support": summary["by_source_reference"],
    })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Inspect trait-classification .eval log")
    parser.add_argument("--output", type=Path, required=True, help="Write report JSON")
    args = parser.parse_args(argv)
    try:
        from inspect_ai.log import read_eval_log
        report = report_from_log(read_eval_log(str(args.log)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote trait report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
