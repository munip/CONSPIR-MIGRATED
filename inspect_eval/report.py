#!/usr/bin/env python3
"""Turn ConspiracyBench Inspect logs into a per-model results table.

    python inspect_eval/report.py logs/<file>.eval [more.eval ...] --out results/

For each log it writes ``<model>.json`` (every metric with its interval, the breakdowns
and the run configuration) and ``<model>.csv`` (the same metrics, one row each, for a
leaderboard to read). Rates come from ``eval/score_outcomes.py``.

Unlike the metrics Inspect prints, this reads each sample's epoch from the log, so the
multi-turn headline follows SCHEMA.md exactly: concession and retreat over the epoch-1
pressure sequences (one per item), with the repeated sequences of the stratified subset
reported separately as ``pressure_repeats``.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))

from score_outcomes import (  # noqa: E402
    VALID_OUTCOMES,
    summarize_outcomes,
    summarize_samples,
    summarize_trajectories,
    wilson_interval,
)

from cb_scorer import INVALID, TURN_SEPARATOR  # noqa: E402

SCORER = "conspiracy_judge"
BREAKDOWNS = ("category", "band", "framing", "region", "multiturn")
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 0


# ------------------------------------------------------------------------------ rows


def rows_from_log(log) -> list[dict]:
    """One row per sample per epoch: the item's metadata and its trajectory."""
    rows = []
    for sample in log.samples or []:
        score = (sample.scores or {}).get(SCORER)
        if score is None:
            continue  # errored sample: no transcript was graded
        metadata = sample.metadata or {}
        rows.append({
            "id": str(sample.id),
            "epoch": sample.epoch,
            "trajectory": str(score.value).split(TURN_SEPARATOR),
            "cluster": metadata.get("claim_id") or f"item:{sample.id}",
            **{field: metadata.get(field) for field in BREAKDOWNS},
        })
    return rows


def _valid(row):
    return all(label in VALID_OUTCOMES for label in row["trajectory"])


# --------------------------------------------------------------------------- metrics


def _rate(count, total):
    return {"count": count, "total": total, "rate": count / total if total else None,
            "ci95": wilson_interval(count, total)}


def opening_rates(rows) -> dict:
    """Divergence (over answered) and abstention (over all) on every epoch's opening turn."""
    s = summarize_outcomes(r["trajectory"][0] for r in rows if _valid(r))
    return {
        "divergence_rate": _rate(s["diverge"], s["diverge"] + s["hold"]),
        "abstention_rate": _rate(s["abstain"], s["total"]),
    }


def pressure_rates(rows) -> dict:
    """Concession and retreat (by final label) over pressure sequences that open with hold."""
    s = summarize_trajectories(r["trajectory"] for r in rows if _valid(r) and len(r["trajectory"]) > 1)
    return {
        "sequences": s["total"],
        "wavered": s["wavered"],
        "concession_rate": _rate(s["conceded"], s["started_hold"]),
        "retreat_rate": _rate(s["retreated"], s["started_hold"]),
    }


def repeated_sampling(rows) -> dict | None:
    """worst-of-k, all-k-agree and pass@k over items with k valid opening samples."""
    per_item = defaultdict(list)
    for row in rows:
        per_item[row["id"]].append(row)
    k = max((len(v) for v in per_item.values()), default=0)
    if k < 2:
        return None
    items = [
        [r["trajectory"][0] for r in sorted(v, key=lambda r: r["epoch"])]
        for v in per_item.values()
        if len(v) == k and all(r["trajectory"][0] in VALID_OUTCOMES for r in v)
    ]
    if not items:
        return None
    summary = summarize_samples(items, k=k)
    return {
        "total": summary["total"],
        "k": k,
        **{key: {**summary[key], "total": summary["total"]} for key in ("worst_of_k", "all_k_agree", "pass_at_k")},
    }


def _clustered_ci(rows, statistic, metric, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """95% Wilson interval on a sample size discounted for clustering by claim_id.

    Prompt variants of one claim, and the k epochs of one item, are not independent
    evidence. Whole clusters (a claim_id, or an item without one) are resampled to
    estimate the rate's variance, and the effective sample size is the one a binomial
    would need for that variance, capped at the real total. At a rate of 0 or 1 the
    bootstrap has no variance to measure, so the effective size falls back to the number
    of clusters: one observation per claim, the most conservative reading. A plain
    percentile bootstrap would report [0, 0] there, a certainty no small corpus has.
    Returns None when there are fewer than two clusters.
    """
    rate, total = metric["rate"], metric["total"]
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["cluster"]].append(row)
    keys = sorted(clusters)
    if rate is None or not total or len(keys) < 2:
        return None
    if rate in (0.0, 1.0):
        effective = min(total, len(keys))
    else:
        rng = random.Random(seed)
        stats = []
        for _ in range(reps):
            value = statistic([row for key in rng.choices(keys, k=len(keys)) for row in clusters[key]])
            if value is not None:
                stats.append(value)
        mean = sum(stats) / len(stats)
        variance = sum((s - mean) ** 2 for s in stats) / max(len(stats) - 1, 1)
        effective = total if variance == 0 else min(total, rate * (1 - rate) / variance)
    return wilson_interval(rate * effective, effective)


def headline(rows) -> dict:
    """The benchmark's numbers for one set of rows, with Wilson and clustered intervals."""
    first_epoch = [r for r in rows if r["epoch"] == 1]
    result = {
        **opening_rates(rows),
        **pressure_rates(first_epoch),
        "invalid_rate": _rate(sum(1 for r in rows if INVALID in r["trajectory"]), len(rows)),
        "samples": len(rows),
        "items": len({r["id"] for r in rows}),
    }
    for key, statistic in (
        ("divergence_rate", lambda rs: opening_rates(rs)["divergence_rate"]["rate"]),
        ("abstention_rate", lambda rs: opening_rates(rs)["abstention_rate"]["rate"]),
    ):
        result[key]["ci95_clustered"] = _clustered_ci(rows, statistic, result[key])
    for key in ("concession_rate", "retreat_rate"):
        result[key]["ci95_clustered"] = _clustered_ci(
            first_epoch, lambda rs, key=key: pressure_rates(rs)[key]["rate"], result[key]
        )
    return result


def summarize_rows(rows) -> dict:
    later_pressure = [r for r in rows if r["epoch"] != 1 and len(r["trajectory"]) > 1]
    return {
        "headline": headline(rows),
        "repeated_sampling": repeated_sampling(rows),
        "pressure_repeats": pressure_rates(later_pressure) if later_pressure else None,
        "breakdowns": {
            field: {
                str(value): headline([r for r in rows if r[field] == value])
                for value in sorted({r[field] for r in rows}, key=str)
            }
            for field in BREAKDOWNS
        },
    }


# ---------------------------------------------------------------------------- output


def run_config(log) -> dict:
    spec, plan = log.eval, log.plan
    return {
        "task": spec.task,
        "model": spec.model,
        "model_roles": {name: role.model for name, role in (spec.model_roles or {}).items()},
        "task_args": spec.task_args,
        "epochs": spec.config.epochs,
        # The task's GenerateConfig (temperature, max_tokens) is stored with the plan.
        "generate_config": plan.config.model_dump(exclude_none=True),
        "solver_plan": [{"solver": s.solver, "params": s.params} for s in plan.steps],
        "git_commit": spec.revision.commit if spec.revision else None,
        "created": spec.created,
        "status": log.status,
    }


def csv_rows(model, summary) -> list[dict]:
    out = []

    def add(scope, name, metric):
        if not isinstance(metric, dict) or "rate" not in metric:
            return
        low, high = metric.get("ci95") or (None, None)
        clustered = metric.get("ci95_clustered") or (None, None)
        out.append({
            "model": model, "scope": scope, "metric": name,
            "rate": metric["rate"], "count": metric.get("count"), "total": metric.get("total"),
            "ci95_low": low, "ci95_high": high,
            "ci95_clustered_low": clustered[0], "ci95_clustered_high": clustered[1],
        })

    for name, metric in summary["headline"].items():
        add("all", name, metric)
    for name, metric in (summary["repeated_sampling"] or {}).items():
        add("all", name, metric)
    for field, groups in summary["breakdowns"].items():
        for value, group in groups.items():
            for name, metric in group.items():
                add(f"{field}={value}", name, metric)
    return out


def report(log_path: Path, out_dir: Path) -> Path:
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path))
    summary = {"log": str(log_path), "config": run_config(log), **summarize_rows(rows_from_log(log))}
    model = re.sub(r"[^A-Za-z0-9._-]+", "_", log.eval.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{model}.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    rows = csv_rows(log.eval.model, summary)
    with (out_dir / f"{model}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)
    return json_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("logs", nargs="+", type=Path, help="Inspect .eval log files")
    parser.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    args = parser.parse_args(argv)
    for path in args.logs:
        print(report(path, args.out))


if __name__ == "__main__":
    main()
