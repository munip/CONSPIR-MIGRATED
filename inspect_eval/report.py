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
from collections import Counter, defaultdict
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

from scorer import INVALID, TURN_SEPARATOR  # noqa: E402

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
            "invalid_reasons": (score.metadata or {}).get("invalid_reasons") or [],
            "truncated_turns": (score.metadata or {}).get("truncated_turns") or [],
            "cluster": metadata.get("claim_id") or f"item:{sample.id}",
            **{field: metadata.get(field) for field in BREAKDOWNS},
        })
    return rows


def root_cause(error) -> str:
    """Why a sample errored, in one line.

    When Inspect stops retrying a model call it raises tenacity's RetryError, whose message
    (``RetryError(<Future at 0x... state=finished raised Transient>)``) hides what kept
    failing. The cause is the first exception in the traceback, so show that.
    """
    message = ((error.message or "").strip().splitlines() or [""])[0]
    if not message.startswith("RetryError"):
        return message[:200]
    for line in (error.traceback or "").splitlines():
        if not line or line[0].isspace() or line.startswith(("Traceback", "The above exception", "During handling")):
            continue
        name, _, detail = line.partition(": ")
        if name.endswith("RetryError"):
            continue  # the wrapper itself, which comes last
        short = name.rsplit(".", 1)[-1]
        return (f"{short}: {detail}" if detail else short)[:180] + " (retries exhausted)"
    return message[:200]


def coverage(log) -> dict:
    """How many samples were planned, completed and errored.

    Rates are computed over the samples that completed, so a run with errors still scores.
    This says how many there were, so it cannot look complete when it is not. An errored sample
    is listed whether or not it was scored: under score_on_error its cut-off transcript is
    graded, with the missing turns marked invalid.
    """
    samples = log.samples or []
    errored = [
        {
            "id": str(s.id),
            "epoch": s.epoch,
            "error": root_cause(s.error),
        }
        for s in samples
        if s.error
    ]
    if log.results:
        # Inspect's own totals, which hold even when the log stores no samples (--no-log-samples).
        planned = log.results.total_samples
        completed = log.results.completed_samples
        errored_count = planned - completed
    else:  # the run stopped early, so count what it was meant to run
        planned = (log.eval.dataset.samples or len(samples)) * (log.eval.config.epochs or 1)
        completed = len(samples) - len(errored)
        errored_count = len(errored)
    # Why the run itself stopped (Inspect's default fail_on_error), or None if it ran to the end.
    run_error = root_cause(log.error) if log.error else None
    if run_error and run_error.startswith("RetryError"):
        # Inspect re-raises the sample's error "from None", so the run's own traceback ends at its
        # scheduler and never names the cause. The sample that stopped the run does; the others it
        # stopped show only as cancelled.
        causes = [e["error"] for e in errored if not e["error"].startswith("CancelledError")]
        run_error = causes[0] if causes else run_error
    return {
        "planned": planned,
        "completed": completed,
        "errored": errored_count,
        "errored_samples": errored,
        "run_error": run_error,
    }


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


def invalid_turns(rows) -> dict:
    """Invalid turns counted by reason, over every turn in the log.

    The headline ``invalid_rate`` counts trajectories with any invalid turn; this counts
    the turns themselves and says why, so a run can be checked against a per-reason bar.
    A turn from a log made before reasons were recorded is counted as ``unrecorded``.
    ``truncated_graded`` counts cut-off replies that were sent to the judge anyway
    (``grade_truncated``), which appear in the rates and would otherwise be invisible.
    """
    total = sum(len(r["trajectory"]) for r in rows)
    by_reason, truncated_graded = Counter(), 0
    for row in rows:
        reasons = row.get("invalid_reasons") or []
        for index, label in enumerate(row["trajectory"]):
            if label == INVALID:
                reason = reasons[index] if index < len(reasons) else None
                by_reason[reason or "unrecorded"] += 1
            elif index in (row.get("truncated_turns") or []):
                truncated_graded += 1
    return {
        "turns": total,
        "invalid": _rate(sum(by_reason.values()), total),
        "by_reason": {reason: _rate(count, total) for reason, count in sorted(by_reason.items())},
        "truncated_graded": _rate(truncated_graded, total),
    }


def summarize_rows(rows) -> dict:
    later_pressure = [r for r in rows if r["epoch"] != 1 and len(r["trajectory"]) > 1]
    return {
        "headline": headline(rows),
        "invalid_turns": invalid_turns(rows),
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
    cover = summary["coverage"]
    add("all", "completion_rate", _rate(cover["completed"], cover["planned"]))
    for name, metric in (summary["repeated_sampling"] or {}).items():
        add("all", name, metric)
    turns = summary["invalid_turns"]
    add("all", "invalid_turn_rate", turns["invalid"])
    for reason, metric in turns["by_reason"].items():
        add("all", f"invalid_turn_rate[{reason}]", metric)
    add("all", "truncated_graded_turn_rate", turns["truncated_graded"])
    for field, groups in summary["breakdowns"].items():
        for value, group in groups.items():
            for name, metric in group.items():
                add(f"{field}={value}", name, metric)
    return out


def report(log_path: Path, out_dir: Path) -> Path:
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path))
    summary = {
        "log": str(log_path),
        "config": run_config(log),
        "coverage": coverage(log),
        **summarize_rows(rows_from_log(log)),
    }
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
        json_path = report(path, args.out)
        print(json_path)
        summary = json.loads(json_path.read_text(encoding="utf-8"))
        cover, turns = summary["coverage"], summary["invalid_turns"]
        if turns["invalid"]["count"] or turns["truncated_graded"]["count"]:
            reasons = ", ".join(f"{m['count']} {reason}" for reason, m in turns["by_reason"].items())
            graded = turns["truncated_graded"]["count"]
            print(
                f"note: {path.name}: {turns['invalid']['count']} of {turns['turns']} turns invalid"
                + (f" ({reasons})" if reasons else "")
                + (f"; {graded} cut-off replies were graded" if graded else "")
                + ".",
                file=sys.stderr,
            )
        if cover["run_error"]:
            print(f"warning: {path.name}: the run stopped early: {cover['run_error']}", file=sys.stderr)
        if cover["errored"] or cover["completed"] < cover["planned"]:
            ids = ", ".join(f"{s['id']} (epoch {s['epoch']})" for s in cover["errored_samples"])
            print(
                f"warning: {path.name}: {cover['completed']} of {cover['planned']} samples completed; "
                f"{cover['errored']} errored{f' ({ids})' if ids else ''}. Rates cover the completed samples only.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
