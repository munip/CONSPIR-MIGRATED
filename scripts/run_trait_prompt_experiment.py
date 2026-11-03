#!/usr/bin/env python3
"""Run the five-fold, four-arm ConspirED prompt/context experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from inspect_ai import eval as inspect_eval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspect_eval.trait.prompt_context_experiment import (  # noqa: E402
    ARMS,
    DEFAULT_BOOTSTRAP_REPETITIONS,
    build_arm_task,
    build_experiment_report,
    build_fold_plan,
    load_experiment_data,
    constant_prediction_rows,
    metric_summary,
    rows_from_eval_log,
)

EXPECTED_MODEL_ID = "hf/Qwen/Qwen2.5-3B-Instruct"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets" / "conspired" / "train.jsonl",
    )
    parser.add_argument("--model", required=True, help="Inspect model identifier")
    parser.add_argument(
        "--model-revision",
        required=True,
        help="Pinned model revision passed to the Inspect model provider",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Model loading dtype; auto uses the dtype declared by the checkpoint",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Maximum number of samples Inspect runs in parallel per arm",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one deterministic positive example through all four arms (four generations)",
    )
    parser.add_argument(
        "--pilot-samples",
        type=int,
        help="Run this many validation examples from fold 1 through all four arms",
    )
    parser.add_argument(
        "--run-environment",
        required=True,
        help="Record hardware, quantization, and runtime/concurrency settings",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "trait-prompt-context",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "logs" / "trait-prompt-context",
    )
    return parser.parse_args()


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _log_location(log: Any) -> str | None:
    location = getattr(log, "location", None)
    return str(location) if location is not None else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.model != EXPECTED_MODEL_ID:
        raise ValueError(f"This experiment is pinned to model {EXPECTED_MODEL_ID!r}")
    if args.folds != 5 or args.seed != 42:
        raise ValueError("This experiment is pinned to five folds with seed 42")
    if args.temperature != 0:
        raise ValueError("This experiment requires --temperature 0")
    if args.max_tasks < 1:
        raise ValueError("--max-tasks must be positive")
    max_samples = getattr(args, "max_samples", 1)
    if max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.bootstrap_repetitions < 1:
        raise ValueError("--bootstrap-repetitions must be positive")
    if not args.model_revision.strip():
        raise ValueError("--model-revision must be nonblank")
    if not args.run_environment.strip():
        raise ValueError("--run-environment must be nonblank")
    smoke_test = getattr(args, "smoke", False)
    pilot_samples = getattr(args, "pilot_samples", None)
    if smoke_test and pilot_samples is not None:
        raise ValueError("--smoke and --pilot-samples cannot be used together")
    if pilot_samples is not None and pilot_samples < 1:
        raise ValueError("--pilot-samples must be positive")
    pilot_test = smoke_test or pilot_samples is not None

    rows, excluded = load_experiment_data(args.dataset)
    dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    plan = build_fold_plan(rows, folds=args.folds, seed=args.seed)
    execution_plan = plan
    pilot_sample_ids: list[str] = []
    if pilot_test:
        execution_plan = copy.deepcopy(plan)
        first_fold = execution_plan["folds"][0]
        validation_ids = first_fold["validation_sample_ids"]
        if smoke_test:
            rows_by_id = {row["id"]: row for row in rows}
            positive_ids = [
                sample_id
                for sample_id in validation_ids
                if any(
                    rows_by_id[sample_id]["trait_annotations"][trait] == "present"
                    for trait in (
                        "contradictory",
                        "overriding_suspicion",
                        "nefarious_intent",
                        "persecuted_victim",
                        "immune_to_evidence",
                        "reinterpreting_randomness",
                    )
                )
            ]
            if not positive_ids:
                raise ValueError("The first fold has no positive example for the smoke test")
            pilot_sample_ids = [random.Random(args.seed).choice(sorted(positive_ids))]
        else:
            if pilot_samples > len(validation_ids):
                raise ValueError(
                    f"--pilot-samples ({pilot_samples}) exceeds the first-fold "
                    f"validation size ({len(validation_ids)})"
                )
            selected_ids = set(random.Random(args.seed).sample(validation_ids, pilot_samples))
            pilot_sample_ids = [
                sample_id for sample_id in validation_ids if sample_id in selected_ids
            ]
        first_fold["validation_sample_ids"] = pilot_sample_ids
        execution_plan["folds"] = [first_fold]

    manifest = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_sha256,
        "sample_count": len(rows),
        "excluded_rows": excluded,
        "fold_plan": plan,
        "execution_plan": execution_plan,
        "smoke_test": smoke_test,
        "pilot_test": pilot_test,
        "pilot_sample_ids": pilot_sample_ids,
        "smoke_sample_ids": pilot_sample_ids if smoke_test else [],
        "arms": {
            arm: {
                "include_context1000": settings[0],
                "include_demonstrations": settings[1],
            }
            for arm, settings in ARMS.items()
        },
        "model": args.model,
        "model_revision": args.model_revision,
        "model_dtype": args.dtype,
        "decoding": "greedy",
        "temperature": args.temperature,
        "max_tasks": args.max_tasks,
        "max_samples": max_samples,
        "run_environment": args.run_environment,
        "bootstrap_repetitions": args.bootstrap_repetitions,
    }
    manifest_path = args.results_dir / "fold_manifest.json"
    report_path = args.results_dir / "report.json"
    existing_results = [path for path in (manifest_path, report_path) if path.exists()]
    if existing_results:
        raise FileExistsError(
            "Experiment outputs already exist: "
            + ", ".join(str(path) for path in existing_results)
        )
    run_log_dirs = [
        args.log_dir / f"fold-{fold_plan['fold']:02d}-{arm}"
        for fold_plan in execution_plan["folds"]
        for arm in ARMS
    ]
    nonempty_log_dirs = [
        path for path in run_log_dirs
        if path.exists() and (not path.is_dir() or any(path.iterdir()))
    ]
    if nonempty_log_dirs:
        raise FileExistsError(
            "Inspect log directories are not empty: "
            + ", ".join(str(path) for path in nonempty_log_dirs)
        )
    _write_manifest(manifest_path, manifest)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    log_locations: dict[str, list[str | None]] = {arm: [] for arm in ARMS}
    for fold_plan in execution_plan["folds"]:
        fold_number = fold_plan["fold"]
        for arm in ARMS:
            task, expected_ids = build_arm_task(rows, execution_plan, fold_number, arm)
            run_log_dir = args.log_dir / f"fold-{fold_number:02d}-{arm}"
            logs = inspect_eval(
                task,
                model=args.model,
                model_args={
                    "revision": args.model_revision,
                    "dtype": args.dtype,
                    "do_sample": False,
                },
                log_dir=str(run_log_dir),
                temperature=None,
                epochs=1,
                sample_shuffle=False,
                fail_on_error=True,
                max_tasks=args.max_tasks,
                max_samples=max_samples,
                display="none",
            )
            if len(logs) != 1:
                raise RuntimeError(
                    f"Expected one Inspect log for fold {fold_number}, arm {arm}; got {len(logs)}"
                )
            log = logs[0]
            status = getattr(log, "status", None)
            status_value = getattr(status, "value", status)
            if status_value not in (None, "success"):
                raise RuntimeError(
                    f"Inspect run failed for fold {fold_number}, arm {arm}: {status_value}"
                )
            scored_rows = rows_from_eval_log(log)
            actual_ids = [row["sample_id"] for row in scored_rows]
            if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
                raise RuntimeError(
                    f"Fold {fold_number}, arm {arm} returned a different sample set "
                    "than its validation partition"
                )
            arm_rows[arm].extend(scored_rows)
            log_locations[arm].append(_log_location(log))

    if pilot_test:
        evaluation_status = (
            "smoke_only_not_a_performance_estimate"
            if smoke_test
            else "pilot_only_not_a_performance_estimate"
        )
        reason = (
            "A one-example smoke test cannot support the experiment's metric or interval criteria."
            if smoke_test
            else "A bounded subset of fold 1 cannot support the experiment's metric or interval criteria."
        )
        report = {
            "arms": {
                arm: metric_summary(rows)
                for arm, rows in arm_rows.items()
            },
            "baselines": {
                "all_six_present": metric_summary(
                    constant_prediction_rows(arm_rows["snippet_only"], predict_all=True)
                ),
                "all_six_absent": metric_summary(
                    constant_prediction_rows(arm_rows["snippet_only"], predict_all=False)
                ),
            },
            "evaluation_status": evaluation_status,
            "primary_comparison": {
                "decision": "not_evaluated_pilot",
                "reason": reason,
            },
        }
    else:
        report = build_experiment_report(
            arm_rows,
            repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        )
    report["run"] = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_sha256,
        "sample_count": len(rows),
        "evaluated_sample_count": len(pilot_sample_ids) if pilot_test else len(rows),
        "excluded_rows": excluded,
        "fold_count": args.folds,
        "seed": args.seed,
        "model": args.model,
        "model_revision": args.model_revision,
        "model_dtype": args.dtype,
        "decoding": "greedy",
        "temperature": args.temperature,
        "max_tasks": args.max_tasks,
        "max_samples": max_samples,
        "smoke_test": smoke_test,
        "pilot_test": pilot_test,
        "pilot_sample_ids": pilot_sample_ids,
        "smoke_sample_ids": pilot_sample_ids if smoke_test else [],
        "run_environment": args.run_environment,
        "inspect_logs": log_locations,
        "fold_manifest": str(manifest_path.resolve()),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return report


def main() -> None:
    args = _arguments()
    report = run(args)
    print(json.dumps(report["primary_comparison"], indent=2))
    print(f"Report: {args.results_dir / 'report.json'}")


if __name__ == "__main__":
    main()
