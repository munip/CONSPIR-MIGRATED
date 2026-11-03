#!/usr/bin/env python3
"""Run the paired, class-balanced ConspirED dominant-trait diagnostic pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspect_eval.trait.dominant_trait_pilot import (  # noqa: E402
    ARMS,
    EXPECTED_MODEL_ID,
    EXPECTED_MODEL_REVISION,
    build_arm_task,
    build_pilot_report,
    load_pilot_data,
    majority_trait,
    rows_from_eval_log,
    select_balanced_sample,
    sha256_file,
    verify_import_manifest,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets" / "conspired-context1000-pilot" / "train.jsonl",
    )
    parser.add_argument("--model", default=EXPECTED_MODEL_ID)
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--items-per-class", type=int, default=10)
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
        default=ROOT / "results" / "dominant-trait-context-pilot",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "logs" / "dominant-trait-context-pilot",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the sample manifest without loading the model or running inference",
    )
    parser.add_argument(
        "--rescore-existing-logs",
        action="store_true",
        help="Recompute scores from saved raw completions without running inference",
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


def _check_output_paths(
    results_dir: Path,
    log_dir: Path,
    prepare_only: bool,
) -> None:
    report_path = results_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(
            "Pilot outputs already exist: "
            + str(report_path)
        )
    manifest_path = results_dir / "sample_manifest.json"
    if prepare_only and manifest_path.exists():
        raise FileExistsError(f"Pilot sample manifest already exists: {manifest_path}")
    if prepare_only:
        return
    arm_log_dirs = [log_dir / arm for arm in ARMS]
    nonempty_log_dirs = [
        path for path in arm_log_dirs
        if path.exists() and (not path.is_dir() or any(path.iterdir()))
    ]
    if nonempty_log_dirs:
        raise FileExistsError(
            "Inspect log directories are not empty: "
            + ", ".join(str(path) for path in nonempty_log_dirs)
        )


def _rescore_existing_logs(args: argparse.Namespace) -> dict[str, Any]:
    import_manifest = verify_import_manifest(args.dataset)
    rows, _ = load_pilot_data(args.dataset)
    manifest_path = args.results_dir / "sample_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pilot sample manifest not found: {manifest_path}")
    sample_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sample_manifest.get("dataset_sha256") != sha256_file(args.dataset):
        raise ValueError("Saved sample manifest does not match the current dataset")
    if sample_manifest.get("source_sha256") != import_manifest["source_sha256"]:
        raise ValueError("Saved sample manifest does not match the import source")
    expected_ids = sample_manifest.get("sample_ids")
    if not isinstance(expected_ids, list) or not all(
        isinstance(sample_id, str) for sample_id in expected_ids
    ):
        raise ValueError("Saved sample manifest has invalid sample IDs")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Saved sample manifest contains duplicate sample IDs")
    items_per_class = sample_manifest.get("items_per_class")
    seed = sample_manifest.get("seed")
    if type(items_per_class) is not int or type(seed) is not int:
        raise ValueError("Saved sample manifest has invalid sampling settings")
    selected = select_balanced_sample(
        rows,
        items_per_class=items_per_class,
        seed=seed,
    )
    if [row["id"] for row in selected] != expected_ids:
        raise ValueError("Saved sample IDs do not match the deterministic sample")
    selected_ids = set(expected_ids)
    baseline = majority_trait(
        [row for row in rows if row["id"] not in selected_ids]
    )
    if sample_manifest.get("development_majority_trait") != baseline:
        raise ValueError("Saved majority baseline does not match the current dataset")

    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    log_locations: dict[str, str] = {}
    for arm in ARMS:
        arm_log_dir = args.log_dir / arm
        log_files = sorted(arm_log_dir.rglob("*.eval")) if arm_log_dir.is_dir() else []
        if len(log_files) != 1:
            raise ValueError(
                f"Expected exactly one saved Inspect log for arm {arm}; "
                f"found {len(log_files)} in {arm_log_dir}"
            )
        log_path = log_files[0]
        log = read_eval_log(log_path)
        status = getattr(log, "status", None)
        status_value = getattr(status, "value", status)
        if status_value not in (None, "success"):
            raise RuntimeError(
                f"Saved Inspect run failed for arm {arm}: {_log_error_message(log)}"
            )
        scored_rows = rows_from_eval_log(log)
        actual_ids = [row["sample_id"] for row in scored_rows]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
            raise ValueError(
                f"Saved log for arm {arm} does not contain exactly the planned sample IDs"
            )
        rows_by_arm[arm] = scored_rows
        log_locations[arm] = str(log_path.resolve())

    report = build_pilot_report(rows_by_arm, expected_ids, baseline)
    report["run"] = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sample_manifest["dataset_sha256"],
        "sample_manifest": str(manifest_path.resolve()),
        "sample_count": len(expected_ids),
        "items_per_class": items_per_class,
        "seed": seed,
        "model": sample_manifest.get("model"),
        "model_revision": sample_manifest.get("model_revision"),
        "model_dtype": sample_manifest.get("model_dtype"),
        "decoding": sample_manifest.get("decoding"),
        "max_tasks": sample_manifest.get("max_tasks"),
        "max_samples": sample_manifest.get("max_samples"),
        "run_environment": sample_manifest.get("run_environment"),
        "inspect_logs": log_locations,
        "rescored_from_existing_logs": True,
        "scoring_note": (
            "Predictions were recomputed from raw Inspect completions; no model "
            "inference was run during rescoring."
        ),
    }
    _write_json_exclusive(args.results_dir / "report_reparsed.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    if getattr(args, "rescore_existing_logs", False):
        return _rescore_existing_logs(args)

    if args.model != EXPECTED_MODEL_ID:
        raise ValueError(f"This pilot is pinned to model {EXPECTED_MODEL_ID!r}")
    if args.model_revision != EXPECTED_MODEL_REVISION:
        raise ValueError(
            f"This pilot is pinned to revision {EXPECTED_MODEL_REVISION!r}"
        )
    if args.seed != 42:
        raise ValueError("This pilot is pinned to --seed 42")
    if args.items_per_class != 10:
        raise ValueError("This pilot is pinned to 10 items per dominant-trait class")
    if args.max_tasks != 1 or args.max_samples != 1:
        raise ValueError("This pilot is pinned to --max-tasks 1 and --max-samples 1")
    if not args.run_environment.strip():
        raise ValueError("--run-environment must be nonblank")

    import_manifest = verify_import_manifest(args.dataset)
    rows, excluded = load_pilot_data(args.dataset)
    selected = select_balanced_sample(
        rows,
        items_per_class=args.items_per_class,
        seed=args.seed,
    )
    selected_ids = {row["id"] for row in selected}
    development_rows = [row for row in rows if row["id"] not in selected_ids]
    baseline = majority_trait(development_rows)

    results_dir = args.results_dir
    log_dir = args.log_dir
    _check_output_paths(results_dir, log_dir, args.prepare_only)
    sample_manifest = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "source_sha256": import_manifest["source_sha256"],
        "context_column_used": import_manifest["context_column_used"],
        "eligible_sample_count": len(rows),
        "excluded_rows": excluded,
        "pilot_sample_count": len(selected),
        "items_per_class": args.items_per_class,
        "seed": args.seed,
        "sample_ids": [row["id"] for row in selected],
        "samples": [
            {
                "id": row["id"],
                "document_id": row["source"]["document_id"],
                "dominant_trait": row["dominant_trait"],
            }
            for row in selected
        ],
        "document_counts": {
            document_id: sum(
                row["source"]["document_id"] == document_id for row in selected
            )
            for document_id in sorted({
                row["source"]["document_id"] for row in selected
            })
        },
        "development_majority_trait": baseline,
        "arms": {
            arm: {"include_context1000": include_context}
            for arm, include_context in ARMS.items()
        },
        "model": args.model,
        "model_revision": args.model_revision,
        "model_dtype": args.dtype,
        "decoding": "greedy",
        "max_tasks": args.max_tasks,
        "max_samples": args.max_samples,
        "run_environment": args.run_environment,
    }
    manifest_path = results_dir / "sample_manifest.json"
    if manifest_path.exists():
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable_manifest = {
            key: value for key, value in saved_manifest.items()
            if key != "prepare_only"
        }
        if comparable_manifest != sample_manifest:
            raise ValueError(
                "Existing sample manifest does not match the dataset, sample, "
                "model, or run settings"
            )
    else:
        _write_json_exclusive(manifest_path, sample_manifest)
    if args.prepare_only:
        return None

    log_dir.mkdir(parents=True, exist_ok=True)
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    log_locations: dict[str, str | None] = {}
    expected_ids = [row["id"] for row in selected]
    for arm in ARMS:
        arm_log_dir = log_dir / arm
        task = build_arm_task(selected, arm)
        logs = inspect_eval(
            task,
            model=args.model,
            model_args={
                "revision": args.model_revision,
                "dtype": args.dtype,
                "do_sample": False,
            },
            log_dir=str(arm_log_dir),
            temperature=None,
            epochs=1,
            sample_shuffle=False,
            fail_on_error=True,
            max_tasks=args.max_tasks,
            max_samples=args.max_samples,
            display="none",
        )
        if len(logs) != 1:
            raise RuntimeError(
                f"Expected one Inspect log for arm {arm}; got {len(logs)}"
            )
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
                f"Arm {arm} returned a different sample set than the pilot manifest"
            )
        rows_by_arm[arm] = scored_rows
        location = getattr(log, "location", None)
        log_locations[arm] = str(location) if location is not None else None

    report = build_pilot_report(rows_by_arm, expected_ids, baseline)
    report["run"] = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sample_manifest["dataset_sha256"],
        "sample_manifest": str(manifest_path.resolve()),
        "sample_count": len(selected),
        "items_per_class": args.items_per_class,
        "seed": args.seed,
        "model": args.model,
        "model_revision": args.model_revision,
        "model_dtype": args.dtype,
        "decoding": "greedy",
        "max_tasks": args.max_tasks,
        "max_samples": args.max_samples,
        "run_environment": args.run_environment,
        "inspect_logs": log_locations,
    }
    _write_json_exclusive(results_dir / "report.json", report)
    return report


def main() -> None:
    args = _arguments()
    report = run(args)
    if report is None:
        print(f"Sample manifest: {args.results_dir / 'sample_manifest.json'}")
        print("Prepared only; no model was loaded or run.")
        return
    print(json.dumps({
        "evaluation_status": report["evaluation_status"],
        "arms": {
            arm: {
                "exact_accuracy": values["exact_accuracy"],
                "valid_single_label_rate": values["valid_single_label_rate"],
            }
            for arm, values in report["arms"].items()
        },
        "context_comparison": report["context_comparison"]["paired_outcomes"],
    }, indent=2))
    report_name = "report_reparsed.json" if args.rescore_existing_logs else "report.json"
    print(f"Report: {args.results_dir / report_name}")


if __name__ == "__main__":
    main()
