#!/usr/bin/env python3
"""Phase B: run the standardized 84-call comparison (#122) with Cloudflare Clef.

Requires run_model_comparison.py from #122. Tasks, case/prompt hashes, scoring
and report come from that harness unchanged; only the model call differs (the
Phase A Clef adapter). Clef's selected choice is stored as raw_output so the
existing label parser and scorer apply as-is.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_baseline as baseline
import run_clef_preflight as clef
import run_model_comparison as cmp
import run_robustness as robustness


def clef_call(
    model: Any,
    processor: Any,
    encode_record: Any,
    systemone: Any,
    task: dict[str, Any],
    model_id: str,
    max_length: int,
) -> dict[str, Any]:
    request = clef.build_clef_request(task, model_id)

    input_tokens = clef.check_no_truncation(
        encode_record,
        processor,
        request,
        max_length,
    )

    started = time.perf_counter()
    response = systemone(model, processor, request, max_length=max_length)
    elapsed = time.perf_counter() - started

    choice, confidence, probabilities = clef.parse_clef_response(response)

    return {
        "raw_output": choice,
        "elapsed_seconds": elapsed,
        "finish_reason": "clef_choice",
        "prompt_tokens": input_tokens,
        "completion_tokens": 0,
        "total_tokens": input_tokens,
        "request_id": None,
        "clef_confidence": confidence,
        "clef_probabilities": probabilities,
        "clef_raw_response": response,
        "retry_count": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Standardized ConspiracyBench judge comparison for Clef."
    )
    ap.add_argument("--repo", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-id", default=clef.DEFAULT_MODEL_ID)
    ap.add_argument("--revision", default=clef.DEFAULT_MODEL_REVISION)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    ap.add_argument("--max-length", type=int, default=16384)

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--resume", action="store_true")

    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    case_path = baseline.find_case_file(repo)
    prompt_path = baseline.find_prompt_file(repo)

    cases = baseline.load_cases(case_path)
    baseline.validate_cases(cases, is_default=True)

    tasks = cmp.build_tasks(cases)
    plan_sha256 = cmp.task_plan_sha256(tasks)

    model_label = f"clef/{args.model_id}@{args.revision[:12]}"

    preflight = {
        "suite_version": cmp.SUITE_VERSION,
        "task_plan_sha256": plan_sha256,
        "model": model_label,
        "provider": "clef",
        "source_case_count": len(cases),
        "core_call_count": 60,
        "stability_repeat_call_count": 24,
        "total_call_count": len(tasks),
        "case_file_sha256": baseline.sha256_file(case_path),
        "prompt_file_sha256": baseline.sha256_file(prompt_path),
        "reasoning_effort": None,
        "temperature": None,
        "max_tokens": None,
        "adapter": "run_clef_preflight.build_clef_request (typed choice)",
        "max_length": args.max_length,
        "prompt_build_errors": [],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    cmp.write_json(out_dir / "comparison_preflight.json", preflight)

    if args.inspect_only:
        print(json.dumps(preflight, indent=2))
        print("Inspection complete. No model was downloaded or called.")
        return 0

    partial_path = out_dir / "comparison_results_partial.jsonl"
    progress_path = out_dir / "comparison_progress.json"

    completed: dict[str, dict[str, Any]] = {}

    if args.run:
        if partial_path.exists() and partial_path.read_text(
            encoding="utf-8"
        ).strip():
            raise SystemExit(
                "Existing partial results found. Use --resume or a new output directory."
            )

    if args.resume:
        if not partial_path.exists() or not progress_path.exists():
            raise SystemExit("Cannot resume: partial/progress files are missing.")

        progress = json.loads(progress_path.read_text(encoding="utf-8"))

        for field, expected in (
            ("suite_version", cmp.SUITE_VERSION),
            ("task_plan_sha256", plan_sha256),
            ("model", model_label),
            ("case_file_sha256", preflight["case_file_sha256"]),
            ("prompt_file_sha256", preflight["prompt_file_sha256"]),
            ("total_call_count", len(tasks)),
        ):
            if field not in progress or progress[field] != expected:
                raise SystemExit(
                    f"Resume provenance mismatch for {field}: "
                    f"{progress.get(field)!r} != {expected!r}"
                )

        for raw in partial_path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                row = json.loads(raw)
                completed[row["task_id"]] = row

    try:
        import torch
        from huggingface_hub import HfApi, snapshot_download

        resolved_revision = HfApi().model_info(
            args.model_id,
            revision=args.revision,
        ).sha

        if not resolved_revision:
            raise RuntimeError("Could not resolve Clef revision to a commit SHA.")

        model_path = snapshot_download(
            repo_id=args.model_id,
            revision=resolved_revision,
        )

        if model_path not in sys.path:
            sys.path.insert(0, model_path)

        from joint_schema_model import (
            encode_record,
            load_release_model,
            systemone,
        )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        model, processor = load_release_model(
            model_path,
            device=args.device,
            dtype=dtype_map[args.dtype],
        )
    except Exception as exc:
        failure = {
            "stage": "model_setup",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        cmp.write_json(out_dir / "comparison_failure.json", failure)
        print(json.dumps(failure, indent=2))
        return 2

    manifest_extra = {
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "suite_commit": clef.git_head(repo),
        "runner_file_sha256": clef.file_sha256(Path(__file__)),
        "adapter_file_sha256": clef.file_sha256(HERE / "run_clef_preflight.py"),
        "runtime": clef.runtime_metadata(torch, args.device, args.dtype),
        "inference_retries": 0,
    }

    try:
        for task in tasks:
            if task["task_id"] in completed:
                continue

            result = clef_call(
                model,
                processor,
                encode_record,
                systemone,
                task,
                args.model_id,
                args.max_length,
            )

            strict, parsed = robustness.parse_labels(result["raw_output"])

            row = {
                **task,
                **result,
                "strict_label": strict,
                "parsed_label": parsed,
                "strict_correct": strict == task["expected_label"],
                "parsed_correct": parsed == task["expected_label"],
                "provider": "clef",
                "model": model_label,
                "reasoning_effort": None,
                "temperature": None,
                "max_tokens": None,
            }

            completed[task["task_id"]] = row

            cmp.write_jsonl(
                partial_path,
                [
                    completed[t["task_id"]]
                    for t in tasks
                    if t["task_id"] in completed
                ],
            )

            cmp.write_json(progress_path, {
                "suite_version": cmp.SUITE_VERSION,
                "task_plan_sha256": plan_sha256,
                "model": model_label,
                "completed": len(completed),
                "total_call_count": len(tasks),
                "case_file_sha256": preflight["case_file_sha256"],
                "prompt_file_sha256": preflight["prompt_file_sha256"],
                "last_completed": task["task_id"],
            })

            print(
                f"[{len(completed)}/{len(tasks)}] "
                f"{task['task_id']}: "
                f"expected={task['expected_label']} "
                f"parsed={parsed} "
                f"confidence={result['clef_confidence']:.4f}",
                flush=True,
            )

    except KeyboardInterrupt:
        print(f"Interrupted after {len(completed)}/{len(tasks)} calls.")
        return 130
    except Exception as exc:
        failure = {
            "stage": "inference",
            "completed_calls": len(completed),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        cmp.write_json(out_dir / "comparison_failure.json", failure)
        print(
            f"Run stopped after {len(completed)}/{len(tasks)} calls: "
            f"{type(exc).__name__}: {exc}"
        )
        return 2

    if len(completed) != len(tasks):
        raise SystemExit(
            f"Expected {len(tasks)} completed calls, found {len(completed)}."
        )

    rows = [completed[t["task_id"]] for t in tasks]

    cmp.write_jsonl(out_dir / "comparison_results.jsonl", rows)

    summary = cmp.summarize(rows, model_label)
    cmp.write_json(out_dir / "comparison_summary.json", summary)
    cmp.write_report(out_dir / "comparison_report.md", summary)

    cmp.write_json(
        out_dir / "comparison_manifest.json",
        {**preflight, **manifest_extra, "completed_calls": len(rows)},
    )

    print()
    print("Comparison completed successfully.")
    print("Summary:", out_dir / "comparison_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
