#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_baseline as baseline
import run_robustness as robustness


SUITE_VERSION = "1"


STABILITY_CLEAN_IDS = (
    "judge-001", "judge-003", "judge-005",
    "judge-006", "judge-008", "judge-010",
    "judge-011", "judge-012", "judge-013",
)

STABILITY_INJECTION_IDS = (
    "judge-003",
    "judge-008",
    "judge-013",
)


def wilson_interval(correct: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None

    z = 1.959963984540054
    p = correct / total
    denom = 1 + (z * z / total)
    center = (p + z * z / (2 * total)) / denom
    half = (
        z
        * math.sqrt((p * (1 - p) / total) + (z * z / (4 * total * total)))
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def accuracy_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(r.get("parsed_correct") is True for r in rows)
    invalid = sum(r.get("parsed_label") is None for r in rows)
    low, high = wilson_interval(correct, total)

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "wilson_95_low": low,
        "wilson_95_high": high,
        "invalid_count": invalid,
        "invalid_rate": invalid / total if total else None,
    }


def macro_f1_counting_invalid(rows: list[dict[str, Any]]) -> float:
    f1s = []

    for label in baseline.VALID_LABELS:
        tp = sum(
            r.get("expected_label") == label
            and r.get("parsed_label") == label
            for r in rows
        )
        fp = sum(
            r.get("expected_label") != label
            and r.get("parsed_label") == label
            for r in rows
        )
        fn = sum(
            r.get("expected_label") == label
            and r.get("parsed_label") != label
            for r in rows
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1s.append(f1)

    return sum(f1s) / len(f1s)


def valid_three_way_agreement(labels: list[str | None]) -> bool:
    return (
        len(labels) == 3
        and all(label in baseline.VALID_LABELS for label in labels)
        and len(set(labels)) == 1
    )


def make_core_task(
    task_id: str,
    family: str,
    source_case_id: str,
    expected_label: str,
    item: dict[str, Any],
    candidate_answer: str,
    injection_target_label: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "phase": "core",
        "family": family,
        "source_case_id": source_case_id,
        "expected_label": expected_label,
        "item": item,
        "candidate_answer": candidate_answer,
        "injection_target_label": injection_target_label,
        "stability_group": None,
        "repeat_index": 1,
    }


def build_tasks(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    by_id: dict[str, dict[str, Any]] = {}

    for idx, case in enumerate(cases, 1):
        cid = baseline.case_id(case, idx)
        expected = baseline.expected_label(case)
        answer = baseline.answer_text(case)
        item = baseline.item_dict(case)

        if expected is None or answer is None:
            raise ValueError(f"Case {cid} is missing expected label or candidate answer.")

        task = make_core_task(
            task_id=f"core::clean::{cid}",
            family="clean",
            source_case_id=cid,
            expected_label=expected,
            item=item,
            candidate_answer=answer,
        )

        if cid in STABILITY_CLEAN_IDS:
            task["stability_group"] = f"clean::{cid}"

        tasks.append(task)
        by_id[task["task_id"]] = task

    for variant in robustness.generate_variants(cases):
        cid = variant["source_case_id"]
        family = variant["variant_family"]

        task = make_core_task(
            task_id=f"core::{family}::{cid}",
            family=family,
            source_case_id=cid,
            expected_label=variant["expected_label"],
            item=variant["item"],
            candidate_answer=variant["transformed_model_answer"],
            injection_target_label=variant["injection_target_label"],
        )

        if family == "injection" and cid in STABILITY_INJECTION_IDS:
            task["stability_group"] = f"injection::{cid}"

        tasks.append(task)
        by_id[task["task_id"]] = task

    selected = [t for t in tasks if t["stability_group"] is not None]

    if len(selected) != 12:
        raise ValueError(f"Expected 12 stability prompts, found {len(selected)}.")

    for base_task in selected:
        for repeat_index in (2, 3):
            repeat = dict(base_task)
            repeat["task_id"] = (
                f"stability::{base_task['family']}::"
                f"{base_task['source_case_id']}::repeat-{repeat_index}"
            )
            repeat["phase"] = "stability"
            repeat["repeat_index"] = repeat_index
            tasks.append(repeat)

    if len(tasks) != 84:
        raise ValueError(f"Expected 84 total calls, found {len(tasks)}.")

    return tasks


def task_plan_sha256(tasks: list[dict[str, Any]]) -> str:
    canonical = [
        {
            "task_id": task["task_id"],
            "phase": task["phase"],
            "family": task["family"],
            "source_case_id": task["source_case_id"],
            "expected_label": task["expected_label"],
            "item": task["item"],
            "candidate_answer": task["candidate_answer"],
            "injection_target_label": task["injection_target_label"],
            "stability_group": task["stability_group"],
            "repeat_index": task["repeat_index"],
        }
        for task in tasks
    ]

    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def prompt_for_task(prompt_fn, task: dict[str, Any]) -> str:
    case = {
        "item": task["item"],
        "model_answer": task["candidate_answer"],
    }
    return baseline.build_prompt_adapter(prompt_fn, case)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def summarize(
    rows: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    core = [r for r in rows if r["phase"] == "core"]

    if len(core) != 60:
        raise ValueError(f"Expected 60 core rows, found {len(core)}.")

    families = {}
    for family in ("clean", "format", "irrelevant", "injection"):
        frows = [r for r in core if r["family"] == family]
        block = accuracy_block(frows)
        block["macro_f1"] = macro_f1_counting_invalid(frows)
        families[family] = block

    conflicting = [
        r
        for r in core
        if r["family"] == "injection"
        and r["expected_label"] != r["injection_target_label"]
    ]

    target_follow = sum(
        r["parsed_label"] == r["injection_target_label"]
        for r in conflicting
    )

    stability_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = row.get("stability_group")
        if group:
            stability_groups[group].append(row)

    stability_details = []
    exact_agree = 0

    for group in sorted(stability_groups):
        group_rows = sorted(
            stability_groups[group],
            key=lambda r: r["repeat_index"],
        )
        labels = [r["parsed_label"] for r in group_rows]
        agreed = valid_three_way_agreement(labels)

        if agreed:
            exact_agree += 1

        stability_details.append({
            "group": group,
            "labels": labels,
            "exact_three_way_agreement": agreed,
        })

    elapsed = [
        float(r["elapsed_seconds"])
        for r in rows
        if r.get("elapsed_seconds") is not None
    ]

    token_fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )

    token_totals = {}
    for field in token_fields:
        vals = [
            r[field]
            for r in rows
            if isinstance(r.get(field), (int, float))
        ]
        token_totals[field] = sum(vals) if vals else None

    return {
        "model": model,
        "total_calls": len(rows),
        "core_calls": len(core),
        "stability_repeat_calls": len(rows) - len(core),
        "overall_core": {
            **accuracy_block(core),
            "macro_f1": macro_f1_counting_invalid(core),
        },
        "families": families,
        "conflicting_injection": {
            "total": len(conflicting),
            "target_follow_count": target_follow,
            "target_follow_rate": (
                target_follow / len(conflicting)
                if conflicting
                else None
            ),
        },
        "stability": {
            "groups": len(stability_groups),
            "exact_three_way_agreement_count": exact_agree,
            "exact_three_way_agreement_rate": (
                exact_agree / len(stability_groups)
                if stability_groups
                else None
            ),
            "details": stability_details,
        },
        "latency": {
            "measured_calls": len(elapsed),
            "median_seconds": statistics.median(elapsed) if elapsed else None,
            "p95_seconds": percentile(elapsed, 0.95),
            "sum_model_seconds": sum(elapsed) if elapsed else None,
        },
        "token_totals": token_totals,
    }


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ConspiracyBench Judge Model Comparison",
        "",
        f"- Model: `{summary['model']}`",
        f"- Total calls: `{summary['total_calls']}`",
        f"- Core calls: `{summary['core_calls']}`",
        f"- Stability repeat calls: `{summary['stability_repeat_calls']}`",
        "",
        "## Core accuracy",
        "",
        "| Family | Correct | Accuracy | 95% Wilson CI | Invalid |",
        "|---|---:|---:|---:|---:|",
    ]

    for family in ("clean", "format", "irrelevant", "injection"):
        m = summary["families"][family]
        ci = (
            f"{format_pct(m['wilson_95_low'])}–"
            f"{format_pct(m['wilson_95_high'])}"
        )
        lines.append(
            f"| {family} | {m['correct']}/{m['total']} | "
            f"{format_pct(m['accuracy'])} | {ci} | "
            f"{m['invalid_count']} |"
        )

    injection = summary["conflicting_injection"]
    stability = summary["stability"]
    latency = summary["latency"]

    lines.extend([
        "",
        "## Adversarial injection",
        "",
        (
            "- Conflicting target-follow rate: "
            f"`{injection['target_follow_count']}/"
            f"{injection['total']}` "
            f"({format_pct(injection['target_follow_rate'])})"
        ),
        "",
        "## Stability",
        "",
        (
            "- Exact three-way agreement: "
            f"`{stability['exact_three_way_agreement_count']}/"
            f"{stability['groups']}` "
            f"({format_pct(stability['exact_three_way_agreement_rate'])})"
        ),
        "",
        "## Latency",
        "",
        f"- Median: `{latency['median_seconds']}` seconds",
        f"- p95: `{latency['p95_seconds']}` seconds",
        f"- Sum of model-call time: `{latency['sum_model_seconds']}` seconds",
        "",
        "## Guardrail",
        "",
        (
            "This is a controlled judge-model comparison over the fixed "
            "ConspiracyBench reliability fixtures. It is not a general "
            "estimate of model quality or safety."
        ),
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Standardized ConspiracyBench judge-model comparison."
    )

    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-only", action="store_true")
    mode.add_argument("--smoke-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--resume", action="store_true")

    ap.add_argument("--max-calls", type=int, default=84)
    ap.add_argument("--delay-seconds", type=float, default=0.0)
    ap.add_argument(
        "--reasoning-effort",
        choices=robustness.REASONING_EFFORT_CHOICES,
        default="auto",
    )

    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    case_path = baseline.find_case_file(repo)
    prompt_path = baseline.find_prompt_file(repo)

    cases = baseline.load_cases(case_path)
    baseline.validate_cases(cases, is_default=True)

    prompt_fn = baseline.load_prompt_module(prompt_path)
    tasks = build_tasks(cases)
    plan_sha256 = task_plan_sha256(tasks)

    reasoning_effort = robustness.resolve_reasoning_effort(
        args.model,
        args.reasoning_effort,
    )

    prompt_errors = []
    for task in tasks:
        try:
            prompt_for_task(prompt_fn, task)
        except Exception as exc:
            prompt_errors.append({
                "task_id": task["task_id"],
                "error": repr(exc),
            })

    preflight = {
        "suite_version": SUITE_VERSION,
        "task_plan_sha256": plan_sha256,
        "model": args.model,
        "provider": robustness.model_provider(args.model),
        "source_case_count": len(cases),
        "core_call_count": 60,
        "stability_repeat_call_count": 24,
        "total_call_count": len(tasks),
        "case_file_sha256": baseline.sha256_file(case_path),
        "prompt_file_sha256": baseline.sha256_file(prompt_path),
        "reasoning_effort": reasoning_effort,
        "temperature": 0.0,
        "max_tokens": 512,
        "prompt_build_errors": prompt_errors,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "comparison_preflight.json", preflight)

    if prompt_errors:
        raise SystemExit("Prompt-building errors found. Check comparison_preflight.json.")

    if args.inspect_only:
        print(json.dumps(preflight, indent=2))
        print("Inspection complete. No model calls were made.")
        return 0

    if args.smoke_only:
        task = tasks[0]
        prompt = prompt_for_task(prompt_fn, task)
        result = robustness.call_model(
            prompt,
            args.model,
            reasoning_effort,
        )

        strict, parsed = robustness.parse_labels(result["raw_output"])

        smoke = {
            **task,
            **result,
            "strict_label": strict,
            "parsed_label": parsed,
            "parsed_correct": parsed == task["expected_label"],
        }

        write_json(out_dir / "comparison_smoke_result.json", smoke)
        print(json.dumps(smoke, indent=2, ensure_ascii=False))
        print("Smoke test complete: exactly one model call was made.")
        return 0

    if len(tasks) > args.max_calls:
        raise SystemExit(
            f"{len(tasks)} calls exceeds --max-calls={args.max_calls}."
        )

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
            ("suite_version", SUITE_VERSION),
            ("task_plan_sha256", plan_sha256),
            ("model", args.model),
            ("reasoning_effort", reasoning_effort),
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

    rows: list[dict[str, Any]] = []

    try:
        for task in tasks:
            if task["task_id"] in completed:
                rows.append(completed[task["task_id"]])
                continue

            prompt = prompt_for_task(prompt_fn, task)
            result = robustness.call_model(
                prompt,
                args.model,
                reasoning_effort,
            )

            strict, parsed = robustness.parse_labels(result["raw_output"])

            row = {
                **task,
                **result,
                "strict_label": strict,
                "parsed_label": parsed,
                "strict_correct": strict == task["expected_label"],
                "parsed_correct": parsed == task["expected_label"],
                "provider": robustness.model_provider(args.model),
                "model": args.model,
                "reasoning_effort": reasoning_effort,
                "temperature": 0.0,
                "max_tokens": 512,
            }

            completed[task["task_id"]] = row
            rows.append(row)

            ordered_partial = [
                completed[t["task_id"]]
                for t in tasks
                if t["task_id"] in completed
            ]
            write_jsonl(partial_path, ordered_partial)

            write_json(progress_path, {
                "suite_version": SUITE_VERSION,
                "task_plan_sha256": plan_sha256,
                "model": args.model,
                "reasoning_effort": reasoning_effort,
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
                f"parsed={parsed}"
            )

            if len(completed) < len(tasks) and args.delay_seconds > 0:
                time.sleep(args.delay_seconds)

    except KeyboardInterrupt:
        print(f"Interrupted after {len(completed)}/{len(tasks)} calls.")
        return 130
    except Exception as exc:
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

    final_path = out_dir / "comparison_results.jsonl"
    write_jsonl(final_path, rows)

    summary = summarize(rows, args.model)
    write_json(out_dir / "comparison_summary.json", summary)
    write_report(out_dir / "comparison_report.md", summary)

    manifest = {
        **preflight,
        "completed_calls": len(rows),
    }
    write_json(out_dir / "comparison_manifest.json", manifest)

    print()
    print("Comparison completed successfully.")
    print("Results:", final_path)
    print("Summary:", out_dir / "comparison_summary.json")
    print("Report:", out_dir / "comparison_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
