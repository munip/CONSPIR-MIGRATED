#!/usr/bin/env python3
"""Run the paired, full ConspirED dominant-trait held-out test evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from inspect_ai import eval as inspect_eval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspect_eval.trait.dominant_trait_pilot import (  # noqa: E402
    ARMS,
    EXPECTED_MODEL_ID,
    EXPECTED_MODEL_REVISION,
    TRAITS,
    _SYSTEM_PROMPT,
    build_arm_task,
    rows_from_eval_log,
    sha256_file,
)
from inspect_eval.trait.dominant_trait_test import (  # noqa: E402
    DEFAULT_BOOTSTRAP_REPETITIONS,
    DEFAULT_BOOTSTRAP_SEED,
    build_test_report,
    training_majority_baseline,
    verify_test_dataset,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    dataset_dir = ROOT / "datasets" / "conspired-context1000-pilot"
    parser.add_argument("--dataset", type=Path, default=dataset_dir / "test.jsonl")
    parser.add_argument("--train-dataset", type=Path, default=dataset_dir / "train.jsonl")
    parser.add_argument("--model", default=EXPECTED_MODEL_ID)
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument(
        "--run-environment",
        required=True,
        help="Record hardware, dtype, quantization, and runtime/concurrency settings",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "dominant-trait-test-evaluation",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "logs" / "dominant-trait-test-evaluation",
    )
    return parser.parse_args()


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _log_error_message(log: Any) -> str:
    error = getattr(log, "error", None)
    if isinstance(error, dict):
        return str(error.get("message", error))
    message = getattr(error, "message", None)
    return str(message if message is not None else error)


def _check_output_paths(args: argparse.Namespace) -> None:
    planned = [
        args.results_dir / "run_manifest.json",
        args.results_dir / "report.json",
    ]
    existing = [path for path in planned if path.exists()]
    if existing:
        raise FileExistsError(
            "Held-out evaluation outputs already exist: "
            + ", ".join(str(path) for path in existing)
        )
    nonempty_log_dirs = [
        args.log_dir / arm
        for arm in ARMS
        if (args.log_dir / arm).exists()
        and (
            not (args.log_dir / arm).is_dir()
            or any((args.log_dir / arm).iterdir())
        )
    ]
    if nonempty_log_dirs:
        raise FileExistsError(
            "Inspect log directories are not empty: "
            + ", ".join(str(path) for path in nonempty_log_dirs)
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.model != EXPECTED_MODEL_ID:
        raise ValueError(f"This evaluation is pinned to model {EXPECTED_MODEL_ID!r}")
    if args.model_revision != EXPECTED_MODEL_REVISION:
        raise ValueError(
            f"This evaluation is pinned to revision {EXPECTED_MODEL_REVISION!r}"
        )
    if args.seed != DEFAULT_BOOTSTRAP_SEED:
        raise ValueError(f"This evaluation is pinned to bootstrap seed {DEFAULT_BOOTSTRAP_SEED}")
    if args.bootstrap_repetitions < 1:
        raise ValueError("--bootstrap-repetitions must be positive")
    if args.max_tasks != 1 or args.max_samples != 1:
        raise ValueError("This evaluation requires --max-tasks 1 and --max-samples 1")
    if not args.run_environment.strip():
        raise ValueError("--run-environment must be nonblank")

    import_manifest, test_rows = verify_test_dataset(args.dataset)
    baseline, baseline_support, training_count = training_majority_baseline(args.train_dataset)
    if (baseline, baseline_support, training_count) != (
        "overriding_suspicion", 697, 1244
    ):
        raise ValueError(
            "Training-majority baseline differs from the predeclared value: "
            f"{baseline} {baseline_support}/{training_count}"
        )
    _check_output_paths(args)

    prompt_sha256 = hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    test_support = {
        trait: sum(row["dominant_trait"] == trait for row in test_rows)
        for trait in TRAITS
    }
    manifest_path = args.results_dir / "run_manifest.json"
    run_manifest = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "source_sha256": import_manifest["source_sha256"],
        "test_record_count": len(test_rows),
        "test_sample_ids": [row["id"] for row in test_rows],
        "test_class_support": test_support,
        "source_document_count": len({
            row["source"]["document_id"] for row in test_rows
        }),
        "train_test_document_overlap": import_manifest["train_test_document_overlap"],
        "context_column_used": import_manifest["context_column_used"],
        "arms": {
            arm: {"include_context1000": include_context}
            for arm, include_context in ARMS.items()
        },
        "model": args.model,
        "model_revision": args.model_revision,
        "model_dtype": args.dtype,
        "decoding": "greedy",
        "system_prompt_sha256": prompt_sha256,
        "baseline": {
            "predicted_trait": baseline,
            "training_support": baseline_support,
            "training_sample_count": training_count,
            "exclusion": "deterministic 10-per-trait pilot sample, seed 42",
        },
        "max_tasks": args.max_tasks,
        "max_samples": args.max_samples,
        "run_environment": args.run_environment,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
    }
    _write_json_exclusive(manifest_path, run_manifest)

    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    log_locations: dict[str, str] = {}
    expected_ids = [row["id"] for row in test_rows]
    for arm in ARMS:
        task = build_arm_task(test_rows, arm)
        log_dir = args.log_dir / arm
        logs = inspect_eval(
            task,
            model=args.model,
            model_args={
                "revision": args.model_revision,
                "dtype": args.dtype,
                "do_sample": False,
            },
            log_dir=str(log_dir),
            temperature=None,
            epochs=1,
            sample_shuffle=False,
            fail_on_error=True,
            max_tasks=args.max_tasks,
            max_samples=args.max_samples,
            display="none",
        )
        if len(logs) != 1:
            raise RuntimeError(f"Expected one Inspect log for arm {arm}; got {len(logs)}")
        log = logs[0]
        status = getattr(log, "status", None)
        status_value = getattr(status, "value", status)
        if status_value not in (None, "success"):
            raise RuntimeError(
                f"Inspect run failed for arm {arm}: {_log_error_message(log)}"
            )
        scored_rows = rows_from_eval_log(log)
        actual_ids = [row["sample_id"] for row in scored_rows]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
            raise RuntimeError(
                f"Arm {arm} did not return exactly all {len(expected_ids)} planned test IDs"
            )
        rows_by_arm[arm] = scored_rows
        location = getattr(log, "location", None)
        if location is None:
            raise RuntimeError(f"Inspect did not provide a saved log location for arm {arm}")
        log_locations[arm] = str(location)

    report = build_test_report(
        rows_by_arm,
        test_rows,
        baseline,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    report["run"] = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": run_manifest["dataset_sha256"],
        "run_manifest": str(manifest_path.resolve()),
        "model": args.model,
        "model_revision": args.model_revision,
        "model_dtype": args.dtype,
        "decoding": "greedy",
        "system_prompt_sha256": prompt_sha256,
        "run_environment": args.run_environment,
        "max_tasks": args.max_tasks,
        "max_samples": args.max_samples,
        "inspect_logs": log_locations,
        "inference_count": len(test_rows) * len(ARMS),
    }
    _write_json_exclusive(args.results_dir / "report.json", report)
    return report


def main() -> None:
    args = _arguments()
    report = run(args)
    print(json.dumps({
        "evaluation_status": report["evaluation_status"],
        "test_sample_count": report["test_sample_count"],
        "source_document_count": report["source_document_count"],
        "arms": {
            arm: {
                "macro_f1": metrics["macro_f1"],
                "exact_accuracy": metrics["exact_accuracy"],
                "valid_single_label_rate": metrics["valid_single_label_rate"],
            }
            for arm, metrics in report["arms"].items()
        },
        "context_comparison": report["context_comparison"],
    }, indent=2))
    print(f"Report: {args.results_dir / 'report.json'}")


if __name__ == "__main__":
    main()
