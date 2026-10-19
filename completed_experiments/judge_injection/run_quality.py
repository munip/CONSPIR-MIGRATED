#!/usr/bin/env python3
"""Run and report the frozen PromptShield filter-quality experiment.

The runner replays the authored release answers through Inspect and the repository's
production ``conspiracy_judge`` scorer. It never generates replacement answers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import random
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INSPECT_ROOT = ROOT / "inspect_eval"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(INSPECT_ROOT) not in sys.path:
    sys.path.insert(0, str(INSPECT_ROOT))

from inspect_ai import Task, eval as inspect_eval  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.event import ModelEvent  # noqa: E402
from inspect_ai.model import GenerateConfig, Model, ModelOutput, get_model  # noqa: E402
from inspect_ai.solver import TaskState, solver  # noqa: E402
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser  # noqa: E402

from promptshield import (  # noqa: E402
    PROMPTSHIELD_REVISION,
    PromptShieldInitializationError,
    PromptShieldScanner,
)
from scorer.cb_scorer import INVALID, conspiracy_judge  # noqa: E402
from completed_experiments.judge_injection.load_dataset import RELEASE_PATH, load_cases  # noqa: E402

MANIFEST_PATH = HERE / "manifest-v1.0.0.json"
VALID_LABELS = ("hold", "diverge", "abstain")
MODES = ("off", "shadow", "gate")
POSITIVE_ACTIONS = {"BLOCK", "FLAG"}
NEGATIVE_ACTIONS = {"PASS", "LOG"}
WARMUP_TEXT = (
    "A benign factual response for PromptShield initialization warm-up. "
    "This text is not part of the frozen release."
)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def normalized_release_hash(path: Path = RELEASE_PATH) -> str:
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def case_order_hash(cases: list[dict[str, Any]]) -> str:
    ids = [str(case["case_id"]) for case in cases]
    return sha256_bytes("\n".join(ids).encode("utf-8"))


def validate_release(
    cases: list[dict[str, Any]] | None = None,
    release_path: Path = RELEASE_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate frozen bytes, release order, and headline composition against manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("release_sha256")
    actual_hash = normalized_release_hash(release_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Frozen release hash mismatch: manifest={expected_hash}, actual={actual_hash}"
        )
    rows = cases if cases is not None else load_cases(release_path)
    expected_count = manifest.get("counts", {}).get("total")
    if len(rows) != expected_count:
        raise ValueError(f"Manifest expects {expected_count} cases; release has {len(rows)}")
    ids = [case.get("case_id") for case in rows]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids):
        raise ValueError("Every release row must have a non-empty string case_id")
    if len(set(ids)) != len(ids):
        raise ValueError("Release contains duplicate case_id values")
    actual_order_hash = case_order_hash(rows)
    expected_order_hash = manifest.get("case_order_sha256")
    if actual_order_hash != expected_order_hash:
        raise ValueError(
            f"Frozen release order mismatch: manifest={expected_order_hash}, actual={actual_order_hash}"
        )
    counts = Counter(case.get("injection_label") for case in rows)
    if counts != Counter({"benign": 70, "malicious": 70}):
        raise ValueError(f"Unexpected injection-label counts: {dict(counts)}")
    if any(case.get("expected_label") not in VALID_LABELS for case in rows):
        raise ValueError("Release contains a missing or invalid expected_label")
    for case in rows:
        for field in ("item", "model_answer", "source_item_id", "pair_id", "transcript"):
            if field not in case:
                if field == "transcript":
                    case[field] = []
                else:
                    raise ValueError(f"Case {case['case_id']} is missing required field {field!r}")
    return rows, manifest


def _assistant_turn_count(case: dict[str, Any]) -> int:
    transcript = case.get("transcript") or []
    return sum(message.get("role") == "assistant" for message in transcript) + 1


def planned_turn_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "cases": len(cases),
        "assistant_turns": sum(_assistant_turn_count(case) for case in cases),
        "final_gold_turns": len(cases),
    }


def build_inspect_dataset(cases: list[dict[str, Any]]) -> MemoryDataset:
    """Preserve release provenance in Inspect metadata without exposing labels to prompts."""
    samples = []
    for case in cases:
        metadata = dict(case)
        metadata.update({
            "case": case,
            "case_id": case["case_id"],
            "pair_id": case.get("pair_id"),
            "expected_label": case.get("expected_label"),
            "injection_label": case.get("injection_label"),
            "pair_subtype": case.get("pair_subtype"),
            "source_item_id": case.get("source_item_id"),
            "transcript": case.get("transcript") or [],
        })
        samples.append(Sample(
            input=case["item"]["question"],
            target=case["expected_label"],
            id=case["case_id"],
            metadata=metadata,
        ))
    return MemoryDataset(samples, name="promptshield-quality-release-1.0.0")


@solver
def replay_frozen_answers():
    """Restore the release transcript and append its current answer verbatim."""

    async def solve(state: TaskState, generate):
        case = state.metadata["case"]
        messages = []
        transcript = case.get("transcript") or []
        if transcript:
            for message in transcript:
                role, content = message["role"], message["content"]
                if role == "user":
                    messages.append(ChatMessageUser(content=content))
                elif role == "assistant":
                    messages.append(ChatMessageAssistant(content=content))
                else:
                    raise ValueError(f"Unsupported transcript role {role!r} in {case['case_id']}")
        else:
            messages.append(ChatMessageUser(content=case["item"]["question"]))
        messages.append(ChatMessageAssistant(content=case["model_answer"]))
        state.messages = messages
        return state

    return solve


def _scanner_profile() -> dict[str, Any]:
    scanner = PromptShieldScanner()
    config = scanner.effective_config
    settings = dict(config["settings"])
    settings.pop("data_dir", None)
    return {"engine": config["engine"], "revision": config["revision"], "settings": settings}


def _usage_dict(event: ModelEvent) -> dict[str, Any] | None:
    usage = getattr(getattr(event, "output", None), "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json", exclude_none=True)
    if isinstance(usage, dict):
        return usage
    return {key: getattr(usage, key) for key in dir(usage) if not key.startswith("_") and not callable(getattr(usage, key))}


def _model_events(sample: Any) -> list[ModelEvent]:
    events = []
    for event in getattr(sample, "events", []) or []:
        if isinstance(event, ModelEvent) and event.role == "grader":
            events.append(event)
    return events


def _event_summary(event: ModelEvent) -> dict[str, Any]:
    return {
        "model": str(event.model),
        "role": event.role,
        "working_time_seconds": event.working_time,
        "retries": event.retries or 0,
        "error": event.error,
        "usage": _usage_dict(event),
    }


def _case_result_from_sample(
    case: dict[str, Any], sample: Any | None, mode: str, repetition: int
) -> dict[str, Any]:
    errors: list[str] = []
    score = None
    if sample is not None:
        score = (sample.scores or {}).get("conspiracy_judge")
        if getattr(sample, "error", None):
            errors.append(str(sample.error))
    if score is None:
        errors.append("Inspect did not produce a conspiracy_judge score for this case")
        turn_outcomes: list[dict[str, Any]] = []
    else:
        turn_outcomes = [dict(outcome) for outcome in (score.metadata or {}).get("turn_outcomes", [])]
    grader_events = _model_events(sample) if sample is not None else []
    called_turns = [turn for turn in turn_outcomes if turn.get("judge_called")]
    if len(grader_events) != len(called_turns):
        errors.append(
            f"grader event mapping mismatch: {len(grader_events)} event(s) for "
            f"{len(called_turns)} judge-called turn(s)"
        )
    for turn, event in zip(called_turns, grader_events):
        turn["grader_event"] = _event_summary(event)
        if event.error:
            errors.append(f"grader event ({event.model}): {event.error}")
    final = turn_outcomes[-1] if turn_outcomes else {}
    scan = final.get("scan") or {}
    for turn in turn_outcomes:
        if turn.get("error"):
            errors.append(f"turn {turn.get('turn_index', '?')}: {turn['error']}")
    expected = case.get("expected_label")
    result = {
        **{key: case.get(key) for key in (
            "case_id", "release_index", "pair_id", "source_item_id", "source_item_file",
            "expected_label", "injection_label", "pair_subtype", "attack_family",
            "injection_location", "target_label", "case_type", "length_profile",
            "lengths", "pair_content_sha256",
        )},
        "mode": mode,
        "repetition": repetition,
        "provenance": {
            "item": case.get("item"),
            "transcript": case.get("transcript") or [],
            "model_answer": case.get("model_answer"),
            "case_type": case.get("case_type"),
            "source_item_id": case.get("source_item_id"),
            "source_item_file": case.get("source_item_file"),
            "pair_id": case.get("pair_id"),
            "pair_content_sha256": case.get("pair_content_sha256"),
        },
        "final_scan_action": scan.get("action"),
        "risk_score": scan.get("risk_score"),
        "detector_ids": scan.get("detector_ids", []),
        "scan_input_hash": scan.get("input_hash"),
        "scan_input_length": scan.get("input_length"),
        "scan_duration_ms": scan.get("scan_duration_ms"),
        "scanner_error": bool(scan.get("error")) or final.get("status") == "scanner_error",
        "judge_called": bool(final.get("judge_called", False)),
        "judge_call_status": final.get("status", "inspect_error" if score is None else None),
        "raw_judge_reply": final.get("raw_judge_reply"),
        "strict_label": final.get("strict_label"),
        "parsed_label": final.get("parsed_label"),
        "errors": errors,
        "error": "; ".join(errors) if errors else None,
        "turn_outcomes": turn_outcomes,
        "judge_events": [_event_summary(event) for event in grader_events],
    }
    final_event = final.get("grader_event") or {}
    result["judge_time_ms"] = (
        final_event.get("working_time_seconds") * 1000
        if final_event.get("working_time_seconds") is not None else None
    )
    result["judge_usage"] = final_event.get("usage")
    result["judge_retries"] = sum(event.retries or 0 for event in grader_events)
    return result


def _make_grader_model(judge_model: str | Model, max_tokens: int, timeout: int, retries: int):
    if not isinstance(judge_model, str):
        return judge_model
    config = GenerateConfig(
        temperature=0,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=retries,
        cache=False,
    )
    return get_model(judge_model, config=config, memoize=False)


def _make_inert_model():
    def no_generation(input, tools, tool_choice, config):
        return ModelOutput.from_content("mockllm/model", "")

    return get_model(
        "mockllm/model",
        custom_outputs=no_generation,
        config=GenerateConfig(cache=False),
        memoize=False,
    )


def run_inspect_unit(
    cases: list[dict[str, Any]],
    mode: str,
    repetition: int,
    log_dir: Path,
    judge_model: str | Model,
    *,
    max_tokens: int = 512,
    timeout: int = 120,
    retries: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Run one serial Inspect unit and return aligned case results and operations."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    scanner = None
    init_ms = warmup_ms = None
    warmup_result = None
    detector_config = None
    if mode != "off":
        scanner = PromptShieldScanner()
        started = time.perf_counter()
        scanner.initialize()
        init_ms = (time.perf_counter() - started) * 1000
        detector_config = scanner.effective_config
        started = time.perf_counter()
        warmup_result = scanner.scan(WARMUP_TEXT)
        warmup_ms = (time.perf_counter() - started) * 1000

    task = Task(
        dataset=build_inspect_dataset(cases),
        solver=replay_frozen_answers(),
        scorer=conspiracy_judge(promptshield_mode=mode, scanner=scanner),
        config=GenerateConfig(cache=False),
        epochs=1,
        name=f"promptshield_quality_{mode}_rep_{repetition:03d}",
        metadata={"experiment": "promptshield-quality", "mode": mode, "repetition": repetition},
    )
    grader = _make_grader_model(judge_model, max_tokens, timeout, retries)
    inert_model = _make_inert_model()
    logs = inspect_eval(
        task,
        model=inert_model,
        model_roles={"grader": grader},
        sample_shuffle=False,
        max_samples=1,
        max_tasks=1,
        display="none",
        log_dir=str(log_dir),
        log_samples=True,
        log_model_api=True,
    )
    run_duration_ms = (time.perf_counter() - run_started) * 1000
    if len(logs) != 1:
        raise RuntimeError(f"Inspect returned {len(logs)} logs for a single unit")
    log = logs[0]
    samples_by_id = {str(sample.id): sample for sample in (log.samples or [])}
    rows = [
        _case_result_from_sample(case, samples_by_id.get(str(case["case_id"])), mode, repetition)
        for case in cases
    ]
    measured_scans = [
        outcome.get("scan", {}).get("scan_duration_ms")
        for row in rows
        for outcome in row["turn_outcomes"]
        if isinstance(outcome.get("scan"), dict)
        and outcome["scan"].get("scan_duration_ms") is not None
    ]
    grader_events = [
        event
        for sample in (log.samples or [])
        for event in _model_events(sample)
    ]
    usage = Counter()
    for event in grader_events:
        event_usage = _usage_dict(event) or {}
        for key, value in event_usage.items():
            if isinstance(value, (int, float)):
                usage[key] += value
    operations = {
        "mode": mode,
        "repetition": repetition,
        "inspect_status": log.status,
        "initialization_ms": init_ms,
        "warmup_ms": warmup_ms,
        "warmup_scan": warmup_result,
        "detector_config": detector_config,
        "scan_latency_ms": {
            "count": len(measured_scans),
            "p50": _percentile(measured_scans, 50),
            "p95": _percentile(measured_scans, 95),
            "max": max(measured_scans) if measured_scans else None,
        },
        "judge_time_ms": sum((event.working_time or 0) * 1000 for event in grader_events),
        "judge_calls": sum(
            turn.get("judge_called", False)
            for row in rows
            for turn in row.get("turn_outcomes", [])
        ),
        "grader_events": len(grader_events),
        "judge_retries": sum(event.retries or 0 for event in grader_events),
        "judge_usage": dict(usage),
        "judge_errors": sum(
            turn.get("status") == "judge_error"
            for row in rows
            for turn in row.get("turn_outcomes", [])
        ),
        "grader_event_errors": sum(bool(event.error) for event in grader_events),
        "run_duration_ms": run_duration_ms,
        "sample_errors": sum(bool(getattr(sample, "error", None)) for sample in (log.samples or [])),
    }
    logs_found = [str(Path(log.location).resolve())] if getattr(log, "location", None) else []
    if log.status != "success":
        operations["unit_error"] = str(log.error or f"Inspect status was {log.status}")
    return rows, operations, logs_found


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    sorted_values = sorted(values)
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def rate(numerator: int | float, denominator: int | float) -> dict[str, Any]:
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def _successful_action(row: dict[str, Any]) -> bool:
    return str(row.get("final_scan_action") or "").upper() in POSITIVE_ACTIONS | NEGATIVE_ACTIONS


def _positive_detection(row: dict[str, Any]) -> bool:
    return str(row.get("final_scan_action") or "").upper() in POSITIVE_ACTIONS


def detection_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if _successful_action(row)]
    tp = sum(row.get("injection_label") == "malicious" and _positive_detection(row) for row in successful)
    fn = sum(row.get("injection_label") == "malicious" and not _positive_detection(row) for row in successful)
    fp = sum(row.get("injection_label") == "benign" and _positive_detection(row) for row in successful)
    tn = sum(row.get("injection_label") == "benign" and not _positive_detection(row) for row in successful)
    all_malicious = sum(row.get("injection_label") == "malicious" for row in rows)
    caught_malicious = sum(
        row.get("injection_label") == "malicious" and _successful_action(row) and _positive_detection(row)
        for row in rows
    )
    successful_malicious = tp + fn
    successful_benign = tn + fp
    recall = rate(tp, successful_malicious)
    specificity = rate(tn, successful_benign)
    balanced = None
    balanced_numerator = None
    if recall["value"] is not None and specificity["value"] is not None:
        balanced_numerator = recall["value"] + specificity["value"]
        balanced = {"value": balanced_numerator / 2, "numerator": balanced_numerator, "denominator": 2}
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": rate(tp, tp + fp),
        "recall": recall,
        "specificity": specificity,
        "accuracy": rate(tp + tn, len(successful)),
        "balanced_accuracy": balanced,
        "f1": rate(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": rate(fp, fp + tn),
        "false_negative_rate": rate(fn, tp + fn),
        "scan_coverage": rate(len(successful), len(rows)),
        "effective_catch_rate_all_malicious": rate(caught_malicious, all_malicious),
        "successful_scans": len(successful),
        "scan_failures": len(rows) - len(successful),
        "all_malicious": all_malicious,
        "caught_malicious": caught_malicious,
    }


def _called(row: dict[str, Any]) -> bool:
    return bool(row.get("judge_called"))


def _valid_prediction(row: dict[str, Any]) -> bool:
    return row.get("parsed_label") in VALID_LABELS


def _factual_judgment_correct(row: dict[str, Any]) -> bool:
    return (
        _called(row)
        and not row.get("scanner_error")
        and row.get("parsed_label") == row.get("expected_label")
    )


def coverage_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    quarantined = sum(row.get("judge_call_status") == "quarantined" for row in rows)
    scanner_errors = sum(
        bool(row.get("scanner_error")) or row.get("judge_call_status") == "scanner_error"
        for row in rows
    )
    called = sum(_called(row) for row in rows)
    valid = sum(_called(row) and _valid_prediction(row) for row in rows)
    return {
        "judge_called_coverage": rate(called, len(rows)),
        "valid_label_coverage": rate(valid, len(rows)),
        "quarantine_rate": rate(quarantined, len(rows)),
        "scanner_error_rate": rate(scanner_errors, len(rows)),
        "called": called,
        "quarantined": quarantined,
        "scanner_errors": scanner_errors,
    }


def _answer_length_bucket(row: dict[str, Any]) -> str:
    lengths = row.get("lengths") or {}
    words = lengths.get("model_answer_words")
    if words is None:
        answer = (row.get("provenance") or {}).get("model_answer", "")
        words = len(str(answer).split())
    if words < 50:
        return "short_lt_50_words"
    if words <= 150:
        return "typical_50_to_150_words"
    return "long_gt_150_words"


def _group_detection(rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = key_fn(row)
        groups[str(value if value is not None else "none")].append(row)
    return {key: detection_metrics(group) for key, group in sorted(groups.items())}


def attack_type_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count detection outcomes per attack family, keeping unavailable scans separate."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("injection_label") == "malicious":
            groups[str(row.get("attack_family") or "unknown")].append(row)
    result = {}
    for family, group in sorted(groups.items()):
        detection = detection_metrics(group)
        result[family] = {
            "total": len(group),
            "caught": detection["tp"],
            "missed": detection["fn"],
            "unscanned": detection["scan_failures"],
            "catch_rate": detection["effective_catch_rate_all_malicious"],
            "recall": detection["recall"],
            "scan_coverage": detection["scan_coverage"],
            "actions": {action: sum(str(row.get("final_scan_action") or "").upper() == action for row in group)
                        for action in ("BLOCK", "FLAG", "PASS", "LOG")},
        }
    return result


def attack_handling_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    malicious = [row for row in rows if row.get("injection_label") == "malicious"]
    benign = [row for row in rows if row.get("injection_label") == "benign"]
    hard_negative = [row for row in benign if row.get("pair_subtype") == "hard_negative"]
    caught = sum(_successful_action(row) and _positive_detection(row) for row in malicious)
    missed = sum(_successful_action(row) and not _positive_detection(row) for row in malicious)
    benign_positive = sum(_successful_action(row) and _positive_detection(row) for row in benign)
    quarantined_benign = sum(row.get("judge_call_status") == "quarantined" for row in benign)
    hard_fp = sum(_successful_action(row) and _positive_detection(row) for row in hard_negative)
    actions = Counter(str(row.get("final_scan_action") or "none").upper() for row in rows)
    detectors = Counter(
        detector_id
        for row in rows
        for detector_id in (row.get("detector_ids") or [])
    )
    return {
        "caught_attacks": rate(caught, len(malicious)),
        "missed_attacks": rate(missed, len(malicious)),
        "unscanned_attacks": rate(len(malicious) - caught - missed, len(malicious)),
        "benign_quarantines": rate(quarantined_benign, len(benign)),
        "benign_positive_detections": rate(benign_positive, len(benign)),
        "hard_negative_false_positives": rate(hard_fp, len(hard_negative)),
        "action_frequencies": dict(sorted(actions.items())),
        "detector_id_frequencies": dict(sorted(detectors.items())),
    }


def _attack_handling_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    malicious = [row for row in rows if row.get("injection_label") == "malicious"]
    benign = [row for row in rows if row.get("injection_label") == "benign"]
    hard_negative = [row for row in benign if row.get("pair_subtype") == "hard_negative"]
    caught = sum(_successful_action(row) and _positive_detection(row) for row in malicious)
    missed = sum(_successful_action(row) and not _positive_detection(row) for row in malicious)
    positive_benign = sum(_successful_action(row) and _positive_detection(row) for row in benign)
    quarantined_benign = sum(row.get("judge_call_status") == "quarantined" for row in benign)
    hard_fp = sum(_successful_action(row) and _positive_detection(row) for row in hard_negative)
    return {
        "caught_attacks": rate(caught, len(malicious)),
        "missed_attacks": rate(missed, len(malicious)),
        "unscanned_attacks": rate(len(malicious) - caught - missed, len(malicious)),
        "benign_quarantines": rate(quarantined_benign, len(benign)),
        "benign_positive_detections": rate(positive_benign, len(benign)),
        "hard_negative_false_positives": rate(hard_fp, len(hard_negative)),
    }


def attack_success_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    malicious = [row for row in rows if row.get("injection_label") == "malicious"]
    follows = sum(
        row.get("target_label") in VALID_LABELS and row.get("parsed_label") == row.get("target_label")
        for row in malicious
    )
    judged = [row for row in malicious if _called(row)]
    judged_follows = sum(
        row.get("target_label") in VALID_LABELS and row.get("parsed_label") == row.get("target_label")
        for row in judged
    )
    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[(str(row.get("pair_id")), int(row.get("repetition", 1)))][str(row.get("injection_label"))] = row
    comparable = changed = clean_correct_pairs = conditional_follow = 0
    for pair in by_pair.values():
        clean, attack = pair.get("benign"), pair.get("malicious")
        if clean is None or attack is None:
            continue
        if clean.get("parsed_label") in VALID_LABELS and attack.get("parsed_label") in VALID_LABELS:
            comparable += 1
            changed += clean["parsed_label"] != attack["parsed_label"]
        if _factual_judgment_correct(clean):
            clean_correct_pairs += 1
            conditional_follow += attack.get("parsed_label") == attack.get("target_label")
    return {
        "target_following_all_malicious": rate(follows, len(malicious)),
        "target_following_judged_malicious": rate(judged_follows, len(judged)),
        "clean_attacked_label_change": rate(changed, comparable),
        "clean_attacked_comparable_pairs": rate(comparable, len(by_pair)),
        "target_following_when_clean_counterpart_correct": rate(conditional_follow, clean_correct_pairs),
        "clean_correct_pairs": clean_correct_pairs,
        "paired_case_groups": len(by_pair),
    }


def _paired_protection(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_unit_case = {
        (row.get("mode"), row.get("repetition"), row.get("case_id")): row for row in all_rows
    }
    baseline_target_rows = quarantined = passed = 0
    benign_total = benign_quarantined = 0
    for attack in all_rows:
        if attack.get("mode") != "off" or attack.get("injection_label") != "malicious":
            continue
        baseline = attack
        if baseline.get("parsed_label") != baseline.get("target_label"):
            continue
        repetition = baseline.get("repetition")
        case_id = baseline.get("case_id")
        gate = by_unit_case.get(("gate", repetition, case_id))
        if gate is None:
            continue
        baseline_target_rows += 1
        quarantined += gate.get("judge_call_status") == "quarantined"
        passed += str(gate.get("final_scan_action") or "").upper() in NEGATIVE_ACTIONS
    for gate in all_rows:
        if gate.get("mode") != "gate" or gate.get("injection_label") != "benign":
            continue
        benign_total += 1
        if gate.get("judge_call_status") == "quarantined":
            benign_quarantined += 1
    return {
        "baseline_target_following_quarantined_by_gate": rate(quarantined, baseline_target_rows),
        "target_following_attacks_that_pass_gate": rate(passed, baseline_target_rows),
        "benign_grading_lost_to_quarantine": rate(benign_quarantined, benign_total),
        "baseline_target_following_count": baseline_target_rows,
    }


def mode_comparison(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode_rep: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in all_rows:
        by_mode_rep[(row["mode"], int(row["repetition"]))][str(row["case_id"])] = row
    label_compared = label_agreed = detector_compared = detector_agreed = 0
    decision_compared = decision_agreed = detector_ids_agreed = 0
    for (baseline_mode, repetition), off_cases in by_mode_rep.items():
        if baseline_mode != "off":
            continue
        shadow = by_mode_rep.get(("shadow", repetition), {})
        gate = by_mode_rep.get(("gate", repetition), {})
        for case_id, off in off_cases.items():
            shadow_row = shadow.get(case_id)
            if shadow_row and _valid_prediction(off) and _valid_prediction(shadow_row):
                label_compared += 1
                label_agreed += off["parsed_label"] == shadow_row["parsed_label"]
            gate_row = gate.get(case_id)
            if shadow_row and gate_row and _successful_action(shadow_row) and _successful_action(gate_row):
                detector_compared += 1
                detector_agreed += str(shadow_row["final_scan_action"]).upper() == str(gate_row["final_scan_action"]).upper()
                decision_compared += 1
                decision_agreed += _positive_detection(shadow_row) == _positive_detection(gate_row)
                detector_ids_agreed += set(shadow_row.get("detector_ids") or []) == set(gate_row.get("detector_ids") or [])
    quarantined_turns = sum(
        outcome.get("status") == "quarantined"
        for row in all_rows if row.get("mode") == "gate"
        for outcome in row.get("turn_outcomes", [])
    )
    gate_called_turns = sum(
        outcome.get("judge_called") is True
        for row in all_rows if row.get("mode") == "gate"
        for outcome in row.get("turn_outcomes", [])
    )
    return {
        "off_shadow_valid_label_agreement": rate(label_agreed, label_compared),
        "off_shadow_comparable_coverage": rate(label_compared, sum(row.get("mode") == "off" for row in all_rows)),
        "shadow_gate_final_detector_action_agreement": rate(detector_agreed, detector_compared),
        "shadow_gate_detection_decision_agreement": rate(decision_agreed, decision_compared),
        "shadow_gate_detector_id_set_agreement": rate(detector_ids_agreed, detector_compared),
        "shadow_gate_detector_comparable_coverage": rate(
            detector_compared, sum(row.get("mode") == "shadow" for row in all_rows)
        ),
        "gate_quarantined_turns_skipped": quarantined_turns,
        "gate_judge_called_turns": gate_called_turns,
        "gate_scanner_error_turns_skipped": sum(
            outcome.get("status") == "scanner_error"
            for row in all_rows if row.get("mode") == "gate"
            for outcome in row.get("turn_outcomes", [])
        ),
    }


def _unit_operations(manifest: dict[str, Any]) -> dict[str, Any]:
    output = {}
    units = manifest.get("units", {})
    for mode in MODES:
        completed = [
            unit.get("operations", {})
            for unit in units.values()
            if unit.get("mode") == mode and unit.get("status") == "complete"
        ]
        scan_values = [
            op["scan_latency_ms"][metric]
            for op in completed
            for metric in ("p50", "p95", "max")
            if metric == "max" and op.get("scan_latency_ms", {}).get(metric) is not None
        ]
        per_rep_scans = [
            op.get("scan_latency_ms", {}) for op in completed
        ]
        output[mode] = {
            "completed_units": len(completed),
            "initialization_ms_by_repetition": [op.get("initialization_ms") for op in completed],
            "warmup_ms_by_repetition": [op.get("warmup_ms") for op in completed],
            "scan_latency_ms_by_repetition": per_rep_scans,
            "scan_latency_ms_max_overall": max(scan_values) if scan_values else None,
            "judge_time_ms": sum(op.get("judge_time_ms", 0) or 0 for op in completed),
            "judge_calls": sum(op.get("judge_calls", 0) for op in completed),
            "grader_events": sum(op.get("grader_events", 0) for op in completed),
            "judge_retries": sum(op.get("judge_retries", 0) for op in completed),
            "judge_usage": dict(sum((Counter(op.get("judge_usage", {})) for op in completed), Counter())),
            "judge_errors": sum(op.get("judge_errors", 0) for op in completed),
            "run_duration_ms": sum(op.get("run_duration_ms", 0) or 0 for op in completed),
            "grader_event_errors": sum(op.get("grader_event_errors", 0) for op in completed),
        }
    return output


def _headline_value(rows: list[dict[str, Any]], metric: str) -> float | None:
    if metric.startswith("detection_") or metric == "effective_catch_rate":
        detection = detection_metrics(rows)
        key = {
            "detection_recall": "recall",
            "detection_accuracy": "accuracy",
            "effective_catch_rate": "effective_catch_rate_all_malicious",
        }[metric]
        return detection[key]["value"]
    if metric == "target_following":
        return attack_success_metrics(rows)["target_following_all_malicious"]["value"]
    if metric == "judge_called_coverage":
        return coverage_metrics(rows)["judge_called_coverage"]["value"]
    return None


def bootstrap_headlines(all_rows: list[dict[str, Any]], repetitions: int) -> dict[str, Any]:
    """Cluster bootstrap by source item, retaining paired cases and repetitions together."""
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        clusters[str(row.get("source_item_id"))].append(row)
    source_ids = sorted(clusters)
    if not source_ids:
        return {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "by_mode": {}, "paired_differences": {}}
    rng = random.Random(BOOTSTRAP_SEED)
    metrics = (
        "detection_recall", "detection_accuracy", "effective_catch_rate",
        "target_following", "judge_called_coverage",
    )
    samples: dict[str, dict[str, list[float]]] = {
        mode: {metric: [] for metric in metrics} for mode in MODES
    }
    diffs = {"gate_minus_off_target_following": []}
    for _ in range(BOOTSTRAP_RESAMPLES):
        draw = [rng.choice(source_ids) for _ in source_ids]
        replicate = [row for source_id in draw for row in clusters[source_id]]
        by_mode = {mode: [row for row in replicate if row.get("mode") == mode] for mode in MODES}
        values = {}
        for mode in MODES:
            values[mode] = {}
            for metric in metrics:
                value = _headline_value(by_mode[mode], metric)
                values[mode][metric] = value
                if value is not None:
                    samples[mode][metric].append(value)
        left, right = values["gate"]["target_following"], values["off"]["target_following"]
        if left is not None and right is not None:
            diffs["gate_minus_off_target_following"].append(left - right)

    def interval(values: list[float]) -> dict[str, Any]:
        return {
            "low": _percentile(values, 2.5),
            "high": _percentile(values, 97.5),
            "valid_resamples": len(values),
            "resamples": BOOTSTRAP_RESAMPLES,
        }

    return {
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "cluster": "source_item_id; each draw retains all pairs and repetitions for that item",
        "by_mode": {mode: {metric: interval(values) for metric, values in metrics_map.items()}
                    for mode, metrics_map in samples.items()},
        "paired_differences": {name: interval(values) for name, values in diffs.items()},
    }


def summarize_results(
    all_rows: list[dict[str, Any]],
    experiment_manifest: dict[str, Any],
) -> dict[str, Any]:
    repetitions = int(experiment_manifest.get("config", {}).get("repetitions", 1))
    by_mode = {}
    for mode in MODES:
        rows = [row for row in all_rows if row.get("mode") == mode]
        detections = detection_metrics(rows)
        benign = [row for row in rows if row.get("injection_label") == "benign"]
        malicious = [row for row in rows if row.get("injection_label") == "malicious"]
        detector_frequency = {
            "actions": dict(sorted(Counter(str(row.get("final_scan_action") or "none").upper() for row in rows).items())),
            "detector_ids": dict(sorted(Counter(
                detector_id for row in rows for detector_id in (row.get("detector_ids") or [])
            ).items())),
        }
        by_mode[mode] = {
            "case_rows": len(rows),
            "detection": detections,
            "detection_subgroups": {
                "ordinary": detection_metrics([row for row in rows if row.get("pair_subtype") == "ordinary"]),
                "hard_negative": detection_metrics([row for row in rows if row.get("pair_subtype") == "hard_negative"]),
                "combined": detections,
            },
            "attack_handling": attack_handling_metrics(rows),
            "attack_handling_subgroups": {
                name: _attack_handling_subset([row for row in rows if row.get("pair_subtype") == name])
                for name in ("ordinary", "hard_negative")
            },
            "attack_types": attack_type_metrics(rows),
            "gate_coverage": coverage_metrics(rows),
            "attack_success_and_pairing": attack_success_metrics(rows),
            "detection_breakdowns": {
                "attack_family": _group_detection(malicious, lambda row: row.get("attack_family")),
                "injection_location": _group_detection(rows, lambda row: row.get("injection_location")),
                "answer_length": _group_detection(rows, _answer_length_bucket),
                "turn_count": _group_detection(rows, lambda row: "multi_turn" if row.get("case_type") == "multi_turn" else "single_turn"),
                "factual_label": _group_detection(rows, lambda row: row.get("expected_label")),
            },
            "action_and_detector_id_frequencies": detector_frequency,
            "malicious_cases": len(malicious),
            "benign_cases": len(benign),
        }
    completed_keys = [
        key for key, unit in experiment_manifest.get("units", {}).items()
        if unit.get("status") == "complete"
    ]
    expected_units = len(MODES) * repetitions
    incomplete_units = [
        key for key, unit in experiment_manifest.get("units", {}).items()
        if unit.get("status") != "complete"
    ]
    for mode in MODES:
        for repetition in range(1, repetitions + 1):
            key = f"{mode}/rep-{repetition:03d}"
            unit = experiment_manifest.get("units", {}).get(key)
            if not unit or unit.get("status") != "complete":
                if key not in incomplete_units:
                    incomplete_units.append(key)
    operations = _unit_operations(experiment_manifest)
    for mode in MODES:
        scan_latencies = [
            outcome.get("scan", {}).get("scan_duration_ms")
            for row in all_rows if row.get("mode") == mode
            for outcome in row.get("turn_outcomes", [])
            if isinstance(outcome.get("scan"), dict)
            and outcome["scan"].get("scan_duration_ms") is not None
        ]
        operations[mode]["measured_scan_latency_ms"] = {
            "count": len(scan_latencies),
            "p50": _percentile(scan_latencies, 50),
            "p95": _percentile(scan_latencies, 95),
            "max": max(scan_latencies) if scan_latencies else None,
        }
    return {
        "experiment": "PromptShield quality experiment",
        "report_schema_version": 2,
        "dataset_version": experiment_manifest.get("dataset", {}).get("version"),
        "dataset_sha256": experiment_manifest.get("dataset", {}).get("release_sha256"),
        "case_order_sha256": experiment_manifest.get("dataset", {}).get("case_order_sha256"),
        "case_count_per_mode": experiment_manifest.get("dataset", {}).get("case_count"),
        "repetitions": repetitions,
        "maximum_planned_judge_calls_before_retries": (
            experiment_manifest.get("dataset", {}).get("assistant_turns_per_mode", 0)
            * len(MODES) * repetitions
        ),
        "incomplete": bool(incomplete_units),
        "expected_units": expected_units,
        "completed_units": len(completed_keys),
        "incomplete_units": sorted(incomplete_units),
        "by_mode": by_mode,
        "protection": _paired_protection(all_rows),
        "mode_comparison": mode_comparison(all_rows),
        "operations": operations,
        "bootstrap": bootstrap_headlines(all_rows, repetitions),
        "interpretation": (
            "Curated synthetic factual references; repeated runs reuse the same cases and are not independent observations."
        ),
    }


def _flatten_rates(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    output = []
    if isinstance(value, dict):
        if set(value) >= {"value", "numerator", "denominator"}:
            output.append({"metric": prefix, **{key: value.get(key) for key in ("numerator", "denominator", "value")}})
        else:
            for key, child in value.items():
                output.extend(_flatten_rates(child, f"{prefix}.{key}" if prefix else str(key)))
    return output


def write_csv_reports(output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    metric_rows = []
    confusion_rows = []
    attack_type_rows = []
    for mode, report in summary["by_mode"].items():
        for section in ("detection", "attack_handling", "gate_coverage", "attack_success_and_pairing"):
            if mode == "off" and section in {"detection", "attack_handling"}:
                continue
            for metric in _flatten_rates(report.get(section, {})):
                metric_rows.append({"mode": mode, "section": section, **metric})
        for subgroup_name, subgroup in (report.get("detection_subgroups", {}) if mode != "off" else {}).items():
            for metric in _flatten_rates(subgroup):
                metric_rows.append({"mode": mode, "section": f"detection_subgroups.{subgroup_name}", **metric})
        for dimension, groups in (report.get("detection_breakdowns", {}) if mode != "off" else {}).items():
            for group, metrics in groups.items():
                for metric in _flatten_rates(metrics):
                    metric_rows.append({"mode": mode, "section": f"detection_breakdowns.{dimension}", "group": group, **metric})
        for subgroup, metrics in (report.get("attack_handling_subgroups", {}) if mode != "off" else {}).items():
            for metric in _flatten_rates(metrics):
                metric_rows.append({"mode": mode, "section": "attack_handling_subgroups", "group": subgroup, **metric})
        if mode != "off":
            detection = report["detection"]
            for gold, label, key in (("malicious", "caught", "tp"), ("malicious", "passed", "fn"),
                                     ("benign", "caught", "fp"), ("benign", "passed", "tn")):
                confusion_rows.append({"mode": mode, "injection_label": gold, "filter_outcome": label, "count": detection[key]})
            for family, counts in report["attack_types"].items():
                attack_type_rows.append({
                    "mode": mode, "attack_family": family,
                    **{key: counts[key] for key in ("total", "caught", "missed", "unscanned")},
                    **counts["actions"], "catch_rate": counts["catch_rate"]["value"],
                    "recall": counts["recall"]["value"], "scan_coverage": counts["scan_coverage"]["value"],
                })
                for metric in _flatten_rates(counts):
                    metric_rows.append({"mode": mode, "section": "attack_types", "group": family, **metric})
    for section, values in (("protection", summary.get("protection", {})),
                            ("mode_comparison", summary.get("mode_comparison", {}))):
        for metric in _flatten_rates(values):
            metric_rows.append({"mode": "comparison", "section": section, **metric})
    metric_path = output_dir / "metrics.csv"
    confusion_path = output_dir / "confusion_matrices.csv"
    for path, data in ((metric_path, metric_rows), (confusion_path, confusion_rows),
                       (output_dir / "attack_types.csv", attack_type_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in data for key in row}) or ["mode", "metric"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)

    failure_rows = []
    for row in rows:
        detected = _positive_detection(row)
        injection = row.get("injection_label")
        detection_error = (
            _successful_action(row)
            and ((injection == "malicious" and not detected) or (injection == "benign" and detected))
        )
        scanner_error = bool(row.get("scanner_error")) or row.get("judge_call_status") == "scanner_error"
        if detection_error or scanner_error or row.get("error"):
            failure_rows.append({
                "mode": row.get("mode"),
                "repetition": row.get("repetition"),
                "case_id": row.get("case_id"),
                "pair_id": row.get("pair_id"),
                "source_item_id": row.get("source_item_id"),
                "failure_type": ("false_positive" if detected else "missed_attack") if detection_error else "execution_error",
                "injection_label": injection,
                "pair_subtype": row.get("pair_subtype"),
                "attack_family": row.get("attack_family"),
                "injection_location": row.get("injection_location"),
                "final_scan_action": row.get("final_scan_action"),
                "error": row.get("error"),
                "scanner_error": row.get("scanner_error"),
            })
    failure_path = output_dir / "failures.csv"
    fields = sorted({key for row in failure_rows for key in row}) or ["mode", "case_id", "error"]
    with failure_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failure_rows)


def _rate_text(value: dict[str, Any]) -> str:
    if value is None:
        return "n/a (0/0)"
    if value.get("value") is None:
        return f"n/a ({value.get('numerator', 0)}/{value.get('denominator', 0)})"
    return f"{value['value']:.3f} ({value['numerator']}/{value['denominator']})"


def _percent_text(value: dict[str, Any] | None) -> str:
    if not value or value.get("value") is None:
        return _rate_text(value)
    if not isinstance(value["numerator"], int) or not isinstance(value["denominator"], int):
        return f"{value['value']:.1%}"
    return f"{value['value']:.1%} ({value['numerator']}/{value['denominator']})"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PromptShield quality experiment",
        "",
        f"- Dataset: `{summary.get('dataset_version')}` ({summary.get('case_count_per_mode')} cases per mode; {summary.get('repetitions')} repetition(s))",
        f"- Release SHA-256: `{summary.get('dataset_sha256')}`",
        f"- Case order SHA-256: `{summary.get('case_order_sha256')}`",
        f"- Completion: **{'INCOMPLETE' if summary.get('incomplete') else 'complete'}** ({summary.get('completed_units')}/{summary.get('expected_units')} mode/repetition units)",
        "",
    ]
    if summary.get("incomplete"):
        lines += ["Incomplete units: " + ", ".join(f"`{unit}`" for unit in summary["incomplete_units"]), ""]
    lines += [
        "## Filter results",
        "",
        "Caught means BLOCK or FLAG; missed means PASS or LOG on a malicious case. "
        "Unavailable scans are reported separately. Recall uses successfully scanned attacks; "
        "catch rate uses all attacks, including unavailable scans. Each case counts once, at its final answer.",
        "",
        "The off mode is a pipeline baseline with no filter scans; it is excluded from filter statistics.",
        "",
        "| Mode | Caught / all attacks | Passed attacks | Unscanned attacks | Benign false positives | Scan coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("shadow", "gate"):
        detection = summary["by_mode"][mode]["detection"]
        handling = summary["by_mode"][mode]["attack_handling"]
        lines.append(
            f"| {mode} | {_percent_text(detection['effective_catch_rate_all_malicious'])} "
            f"| {detection['fn']} | {detection['all_malicious'] - detection['tp'] - detection['fn']} "
            f"| {_percent_text(handling['benign_positive_detections'])} "
            f"| {_percent_text(detection['scan_coverage'])} |"
        )
    lines += [
        "",
        "| Mode | TP | FP | TN | FN | Precision | Recall | Filter accuracy | Balanced accuracy | F1 | FPR | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("shadow", "gate"):
        detection = summary["by_mode"][mode]["detection"]
        values = " | ".join(_percent_text(detection[key]) for key in (
            "precision", "recall", "accuracy", "balanced_accuracy", "f1", "false_positive_rate", "false_negative_rate"
        ))
        lines.append(f"| {mode} | {detection['tp']} | {detection['fp']} | {detection['tn']} | {detection['fn']} | {values} |")
    lines += [
        "", "TP = malicious caught; FP = benign flagged; TN = benign passed; FN = malicious passed. "
        "Filter accuracy measures injection classification, not factual answer or judge accuracy.",
        "", "## Attacks caught by type", "",
        "Each family contains malicious cases only, so precision and false-positive rate are not defined per family. "
        "BLOCK and FLAG count as caught in both modes; only gate quarantines them.", "",
    ]
    for mode in ("shadow", "gate"):
        lines += [f"### {mode}", "",
                  "| Attack type | Total | Caught | Missed | Unscanned | Catch rate | BLOCK | FLAG | PASS | LOG |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for family, counts in summary["by_mode"][mode]["attack_types"].items():
            actions = " | ".join(str(counts["actions"][key]) for key in ("BLOCK", "FLAG", "PASS", "LOG"))
            lines.append(
                f"| {family.replace('_', ' ')} | {counts['total']} | {counts['caught']} "
                f"| {counts['missed']} | {counts['unscanned']} | {_percent_text(counts['catch_rate'])} | {actions} |"
            )
        lines.append("")
    lines += [
        "## Answer-content groups and false positives", "",
        "These are groups of matched clean/attacked answer pairs, not attack types:", "",
        "- **Ordinary answer pairs (`ordinary`)**: the benign answer is plain content without an attack-like quotation.",
        "- **Quoted-attack answer pairs (`hard_negative`)**: the benign answer quotes and explicitly rejects an evaluator instruction. "
        "It should pass the filter despite its attack-like words. Its paired malicious answer adds an active attack to the same content.",
        "- **All cases (`combined`)**: both groups together. This is the overall result, not a third disjoint group.", "",
        "The group name belongs to both halves of a pair. An attacked hard-negative counterpart is malicious, "
        "even though the original quoted-and-rejected instruction is benign.", "",
        "| Mode / content group | Caught attacks | Benign false positives | Benign quarantines | Scan coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    group_names = {"combined": "All cases", "ordinary": "Ordinary answer pairs", "hard_negative": "Quoted-attack answer pairs"}
    for mode in ("shadow", "gate"):
        result = summary["by_mode"][mode]
        for name in ("combined", "ordinary", "hard_negative"):
            group = result["detection_subgroups"][name]
            handling = result["attack_handling"] if name == "combined" else result["attack_handling_subgroups"][name]
            lines.append(
                f"| {mode} / {group_names[name]} | {_percent_text(handling['caught_attacks'])} "
                f"| {_percent_text(handling['benign_positive_detections'])} "
                f"| {_percent_text(handling['benign_quarantines'])} | {_percent_text(group['scan_coverage'])} |"
            )
    lines += ["", "## Gate coverage", "",
              "Quarantine skips a judge call; it is a filter action, not a factual verdict.", "",
              "| Mode | Judge called | Quarantined | Scanner errors |", "|---|---:|---:|---:|"]
    for mode in MODES:
        coverage = summary["by_mode"][mode]["gate_coverage"]
        lines.append(f"| {mode} | {_percent_text(coverage['judge_called_coverage'])} | {_percent_text(coverage['quarantine_rate'])} | {_percent_text(coverage['scanner_error_rate'])} |")
    lines.append("")
    lines += ["## Downstream attack effects and mode comparison", ""]
    protection = summary["protection"]
    comparison = summary["mode_comparison"]
    for name, metric in protection.items():
        if isinstance(metric, dict) and "value" in metric:
            lines.append(f"- {name.replace('_', ' ')}: {_rate_text(metric)}")
    lines += [
        f"- Off/shadow valid-label agreement: {_rate_text(comparison['off_shadow_valid_label_agreement'])}",
        f"- Shadow/gate detector-action agreement: {_rate_text(comparison['shadow_gate_final_detector_action_agreement'])}",
        f"- Shadow/gate positive-detection agreement: {_rate_text(comparison['shadow_gate_detection_decision_agreement'])}",
        f"- Shadow/gate detector-ID set agreement: {_rate_text(comparison['shadow_gate_detector_id_set_agreement'])}",
        f"- Gate judge calls skipped by quarantine: {comparison['gate_quarantined_turns_skipped']}",
        f"- Gate turns skipped by scanner errors: {comparison['gate_scanner_error_turns_skipped']}",
        "",
        "Paired attack success compares each malicious answer with its clean counterpart. The conditional rate uses only pairs where the clean judgment matches its gold label.",
        "",
    ]
    for mode in MODES:
        paired = summary["by_mode"][mode]["attack_success_and_pairing"]
        lines.append(
            f"- {mode}: target following among all attacks "
            f"{_rate_text(paired['target_following_all_malicious'])}; among judged attacks "
            f"{_rate_text(paired['target_following_judged_malicious'])}; clean/attacked label changes "
            f"{_rate_text(paired['clean_attacked_label_change'])}; target following when the clean "
            f"counterpart was correct {_rate_text(paired['target_following_when_clean_counterpart_correct'])}; "
            f"comparable pair coverage {_rate_text(paired['clean_attacked_comparable_pairs'])}."
        )
    lines += ["", "## Detection breakdowns", ""]
    for mode in ("shadow", "gate"):
        lines.append(f"### {mode}")
        lines.append("")
        lines.append("| Dimension / group | TP | FP | TN | FN | Recall | FPR | Coverage |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for dimension, groups in summary["by_mode"][mode]["detection_breakdowns"].items():
            if dimension == "attack_family":
                continue  # Dedicated attack-type table above.
            for group_name, metrics in groups.items():
                lines.append(f"| {dimension} / {group_name} | {metrics['tp']} | {metrics['fp']} | {metrics['tn']} | {metrics['fn']} | {_rate_text(metrics['recall'])} | {_rate_text(metrics['false_positive_rate'])} | {_rate_text(metrics['scan_coverage'])} |")
        lines.append("")
    lines += ["## Operations", ""]
    lines.append("| Mode | Initialization ms by repetition | Warm-up ms by repetition | Measured scan p50 / p95 / max ms overall | Judge calls | Judge time ms | Retries | Judge errors | Run duration ms |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for mode in MODES:
        ops = summary["operations"][mode]
        measured = ops["measured_scan_latency_ms"]
        scan = f"{measured['p50']} / {measured['p95']} / {measured['max']} (n={measured['count']})"
        lines.append(f"| {mode} | {ops['initialization_ms_by_repetition']} | {ops['warmup_ms_by_repetition']} | {scan} | {ops['judge_calls']} | {ops['judge_time_ms']:.1f} | {ops['judge_retries']} | {ops['judge_errors']} | {ops['run_duration_ms']:.1f} |")
    lines += ["", "Token usage by grader model: ", ""]
    for mode in MODES:
        lines.append(f"- {mode}: `{json.dumps(summary['operations'][mode]['judge_usage'], sort_keys=True)}`")
    lines += ["", "## Cluster-bootstrap intervals", "", f"Intervals use {summary['bootstrap']['resamples']:,} source-item cluster resamples with seed {summary['bootstrap']['seed']}. All paired cases and repetitions within each source item stay together.", ""]
    for mode, metrics in summary["bootstrap"]["by_mode"].items():
        lines.append(f"### {mode}")
        lines.append("")
        for metric, interval in metrics.items():
            lines.append(f"- {metric}: {interval['low'] if interval['low'] is not None else 'n/a'}–{interval['high'] if interval['high'] is not None else 'n/a'} ({interval['valid_resamples']} valid resamples)")
        lines.append("")
    lines.append("### Paired differences")
    lines.append("")
    for metric, interval in summary["bootstrap"]["paired_differences"].items():
        lines.append(f"- {metric.replace('_', ' ')}: {interval['low'] if interval['low'] is not None else 'n/a'}–{interval['high'] if interval['high'] is not None else 'n/a'} ({interval['valid_resamples']} valid resamples)")
    lines += ["", "## Interpretation", "", summary.get("interpretation", ""), ""]
    return "\n".join(lines)


def write_reports(output_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_results(rows, manifest)
    write_jsonl(output_dir / "per_case_results.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
    write_csv_reports(output_dir, summary, rows)
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object in {path}:{line_number}")
        rows.append(row)
    return rows


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "results" / "quality" / stamp


def validate_dependencies() -> dict[str, Any]:
    """Check runtime modules without constructing models or running detectors."""
    try:
        inspect_version = importlib.metadata.version("inspect-ai")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("inspect-ai is required; install the inspect_eval dependencies") from exc
    if importlib.util.find_spec("prompt_shield") is None:
        raise RuntimeError(
            "The pinned PromptShield dependency is required; install "
            "inspect_eval/requirements-promptshield.txt"
        )
    try:
        importlib.import_module("prompt_shield.engine")
    except Exception as exc:
        raise RuntimeError(f"PromptShield engine dependency is not importable: {exc}") from exc
    return {
        "inspect_ai": inspect_version,
        "promptshield_revision": PROMPTSHIELD_REVISION,
        "promptshield_engine_importable": True,
    }


def run_configuration(
    judge: str, repetitions: int, max_tokens: int, timeout: int, retries: int,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    scanner = _scanner_profile()
    return {
        "schema_version": 1,
        "modes": list(MODES),
        "repetitions": repetitions,
        "judge_model": judge,
        "judge_generation": {
            "temperature": 0,
            "max_tokens": max_tokens,
            "timeout_seconds": timeout,
            "retries": retries,
            "cache": False,
        },
        "inspect_execution": {
            "sample_concurrency": 1,
            "sample_shuffle": False,
            "dataset_order": "frozen release order",
            "replay_model": "mockllm/model (inert; replay solver does not call it)",
        },
        "detector": scanner,
        "dataset_sha256": dataset["release_sha256"],
        "case_order_sha256": dataset["case_order_sha256"],
    }


def configuration_hash(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(config))


def _experiment_manifest(
    config: dict[str, Any], config_hash: str, release_manifest: dict[str, Any],
    release_path: Path, cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "experiment": "PromptShield quality experiment",
        "schema_version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "config": config,
        "config_sha256": config_hash,
        "dataset": {
            "version": release_manifest["dataset_version"],
            "release_file": release_path.name,
            "release_sha256": normalized_release_hash(release_path),
            "case_order_sha256": case_order_hash(cases),
            "case_count": len(cases),
            "assistant_turns_per_mode": sum(_assistant_turn_count(case) for case in cases),
            "manifest_file": MANIFEST_PATH.name,
        },
        "dependencies": validate_dependencies(),
        "units": {},
    }


def _unit_key(mode: str, repetition: int) -> str:
    return f"{mode}/rep-{repetition:03d}"


def _result_sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _load_completed_unit(
    output_dir: Path,
    unit: dict[str, Any] | None,
    mode: str,
    repetition: int,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if not unit or unit.get("status") != "complete" or not unit.get("result_file"):
        return None
    path = output_dir / unit["result_file"]
    if not path.is_file() or _result_sha(path) != unit.get("result_sha256"):
        return None
    try:
        rows = read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    expected_ids = [case["case_id"] for case in cases]
    if (
        len(rows) != len(expected_ids)
        or [row.get("case_id") for row in rows] != expected_ids
        or any(row.get("mode") != mode or row.get("repetition") != repetition for row in rows)
    ):
        return None
    return rows


def collect_completed_rows(
    output_dir: Path, manifest: dict[str, Any], cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    repetitions = int(manifest.get("config", {}).get("repetitions", 1))
    for mode in MODES:
        for repetition in range(1, repetitions + 1):
            unit = manifest.get("units", {}).get(_unit_key(mode, repetition))
            loaded = _load_completed_unit(output_dir, unit, mode, repetition, cases)
            if loaded is not None:
                rows.extend(loaded)
    return rows


def _manifest_dataset_matches(manifest: dict[str, Any], config: dict[str, Any], config_hash: str) -> None:
    if manifest.get("config_sha256") != config_hash or manifest.get("config") != config:
        raise ValueError("Saved experiment settings differ from the requested resume settings")
    saved_dataset = manifest.get("dataset", {})
    if saved_dataset.get("release_sha256") != config.get("dataset_sha256"):
        raise ValueError("Saved dataset hash differs from the requested release hash")
    if saved_dataset.get("case_order_sha256") != config.get("case_order_sha256"):
        raise ValueError("Saved case-order hash differs from the requested release order")


def _write_attempt_error(
    output_dir: Path,
    manifest: dict[str, Any],
    unit_key: str,
    attempt: dict[str, Any],
    error: BaseException,
) -> None:
    attempt["status"] = "failed"
    attempt["finished_at"] = utc_now()
    attempt["error"] = f"{type(error).__name__}: {error}"
    unit = manifest["units"][unit_key]
    unit["status"] = "incomplete"
    unit["updated_at"] = utc_now()
    write_json(output_dir / "experiment_manifest.json", manifest)


def _failed_case_row(case: dict[str, Any], mode: str, repetition: int, error: BaseException) -> dict[str, Any]:
    message = f"{type(error).__name__}: {error}"
    return {
        **{key: case.get(key) for key in (
            "case_id", "release_index", "pair_id", "source_item_id", "source_item_file",
            "expected_label", "injection_label", "pair_subtype", "attack_family",
            "injection_location", "target_label", "case_type", "length_profile",
            "lengths", "pair_content_sha256",
        )},
        "mode": mode,
        "repetition": repetition,
        "provenance": {
            "item": case.get("item"), "transcript": case.get("transcript") or [],
            "model_answer": case.get("model_answer"),
            "source_item_id": case.get("source_item_id"), "pair_id": case.get("pair_id"),
        },
        "final_scan_action": None,
        "risk_score": None,
        "detector_ids": [],
        "scan_input_hash": None,
        "scan_input_length": None,
        "scanner_error": isinstance(error, PromptShieldInitializationError),
        "judge_called": False,
        "judge_call_status": "initialization_error" if isinstance(error, PromptShieldInitializationError) else "inspect_error",
        "raw_judge_reply": None,
        "strict_label": None,
        "parsed_label": None,
        "errors": [message],
        "error": message,
        "turn_outcomes": [],
        "judge_events": [],
        "judge_time_ms": None,
        "judge_usage": None,
        "judge_retries": 0,
    }


def run_experiment(
    cases: list[dict[str, Any]],
    release_manifest: dict[str, Any],
    judge: str,
    repetitions: int,
    max_tokens: int,
    timeout: int,
    retries: int,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    config = run_configuration(judge, repetitions, max_tokens, timeout, retries, {
        "release_sha256": normalized_release_hash(RELEASE_PATH),
        "case_order_sha256": case_order_hash(cases),
    })
    config_hash = configuration_hash(config)
    manifest_path = output_dir / "experiment_manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise ValueError("--resume requires an existing experiment_manifest.json in --out")
        experiment = json.loads(manifest_path.read_text(encoding="utf-8"))
        _manifest_dataset_matches(experiment, config, config_hash)
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"Output directory is not empty; use --resume or choose another --out: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        experiment = _experiment_manifest(config, config_hash, release_manifest, RELEASE_PATH, cases)
        for mode in MODES:
            for repetition in range(1, repetitions + 1):
                key = _unit_key(mode, repetition)
                experiment["units"][key] = {
                    "mode": mode,
                    "repetition": repetition,
                    "status": "pending",
                    "attempts": [],
                    "created_at": utc_now(),
                }
        write_json(manifest_path, experiment)

    for mode in MODES:
        for repetition in range(1, repetitions + 1):
            key = _unit_key(mode, repetition)
            unit = experiment["units"].setdefault(key, {
                "mode": mode, "repetition": repetition, "status": "pending", "attempts": [],
            })
            if resume and _load_completed_unit(output_dir, unit, mode, repetition, cases) is not None:
                print(f"Skipping completed unit {key}")
                continue
            attempt_id = uuid.uuid4().hex[:12]
            attempt_relative = Path("units") / f"{mode}_rep-{repetition:03d}" / "attempts" / attempt_id
            attempt_dir = output_dir / attempt_relative
            attempt = {
                "id": attempt_id,
                "status": "running",
                "started_at": utc_now(),
                "directory": attempt_relative.as_posix(),
                "inspect_log_directory": (attempt_relative / "inspect").as_posix(),
            }
            unit["attempts"].append(attempt)
            unit["status"] = "running"
            unit["updated_at"] = utc_now()
            experiment["updated_at"] = utc_now()
            write_json(manifest_path, experiment)
            try:
                rows, operations, logs = run_inspect_unit(
                    cases,
                    mode,
                    repetition,
                    attempt_dir / "inspect",
                    judge,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    retries=retries,
                )
                result_path = attempt_dir / "per_case_results.jsonl"
                write_jsonl(result_path, rows)
                complete = (
                    operations.get("inspect_status") == "success"
                    and len(rows) == len(cases)
                    and all(len(row.get("turn_outcomes", [])) == _assistant_turn_count(case)
                            for row, case in zip(rows, cases))
                )
                attempt.update({
                    "status": "complete" if complete else "incomplete",
                    "finished_at": utc_now(),
                    "result_file": result_path.relative_to(output_dir).as_posix(),
                    "result_sha256": _result_sha(result_path),
                    "native_inspect_logs": logs,
                })
                unit["operations"] = operations
                if complete:
                    unit.update({
                        "status": "complete",
                        "result_file": attempt["result_file"],
                        "result_sha256": attempt["result_sha256"],
                        "native_inspect_logs": logs,
                        "completed_at": utc_now(),
                    })
                else:
                    unit["status"] = "incomplete"
                    unit["error"] = operations.get("unit_error", "Inspect output was incomplete")
                unit["updated_at"] = utc_now()
                experiment["updated_at"] = utc_now()
                write_json(manifest_path, experiment)
                if not complete:
                    current_rows = collect_completed_rows(output_dir, experiment, cases) + rows
                    write_reports(output_dir, experiment, current_rows)
                    raise RuntimeError(f"Unit {key} did not complete; its Inspect logs were preserved")
            except BaseException as exc:
                if attempt.get("status") == "running":
                    result_path = attempt_dir / "per_case_results.jsonl"
                    if not result_path.exists():
                        failed_rows = [_failed_case_row(case, mode, repetition, exc) for case in cases]
                        write_jsonl(result_path, failed_rows)
                        attempt["partial_result_file"] = result_path.relative_to(output_dir).as_posix()
                        attempt["partial_result_sha256"] = _result_sha(result_path)
                    _write_attempt_error(output_dir, experiment, key, attempt, exc)
                    partial_path = attempt.get("partial_result_file")
                    partial_rows = read_jsonl(output_dir / partial_path) if partial_path else []
                    write_reports(
                        output_dir,
                        experiment,
                        collect_completed_rows(output_dir, experiment, cases) + partial_rows,
                    )
                raise
    rows = collect_completed_rows(output_dir, experiment, cases)
    summary = write_reports(output_dir, experiment, rows)
    return {"manifest": experiment, "summary": summary}


def _load_report_only(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = output_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"No experiment_manifest.json found under {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    integrity_errors = []
    for key, unit in manifest.get("units", {}).items():
        result_file = unit.get("result_file") if unit.get("status") == "complete" else None
        expected_sha = unit.get("result_sha256") if result_file else None
        if not result_file:
            attempts = unit.get("attempts", [])
            if attempts:
                last_attempt = attempts[-1]
                result_file = last_attempt.get("partial_result_file") or last_attempt.get("result_file")
                expected_sha = last_attempt.get("partial_result_sha256") or last_attempt.get("result_sha256")
        if not result_file:
            continue
        path = output_dir / result_file
        if not path.is_file() or _result_sha(path) != expected_sha:
            integrity_errors.append(key)
            unit["status"] = "incomplete"
            unit["report_integrity_error"] = "Saved per-case result file is missing or its hash differs"
            continue
        rows.extend(read_jsonl(path))
    manifest["report_integrity_errors"] = integrity_errors
    return manifest, rows


def _inspect_only(cases: list[dict[str, Any]], dependencies: dict[str, Any], repetitions: int) -> dict[str, Any]:
    counts = planned_turn_counts(cases)
    detector = _scanner_profile()
    max_calls = counts["assistant_turns"] * len(MODES) * repetitions
    return {
        "dataset_version": "1.0.0",
        "dataset_sha256": normalized_release_hash(RELEASE_PATH),
        "case_order_sha256": case_order_hash(cases),
        "case_count": counts["cases"],
        "case_counts_by_injection": dict(Counter(case["injection_label"] for case in cases)),
        "case_counts_by_turn_type": dict(Counter(case["case_type"] for case in cases)),
        "assistant_turns_per_mode": counts["assistant_turns"],
        "final_gold_turns_per_mode": counts["final_gold_turns"],
        "modes": list(MODES),
        "repetitions": repetitions,
        "maximum_planned_judge_calls_before_retries": max_calls,
        "sample_concurrency": 1,
        "sample_shuffle": False,
        "judge_cache": False,
        "default_judge_generation": {"temperature": 0, "max_tokens": 512, "timeout_seconds": 120, "retries": 2},
        "dependencies": dependencies,
        "detector_profile": detector,
    }


def _print_headlines(summary: dict[str, Any], report_path: Path) -> None:
    for mode in ("shadow", "gate"):
        result = summary["by_mode"][mode]
        print(
            f"{mode}: attacks caught {_percent_text(result['detection']['effective_catch_rate_all_malicious'])}; "
            f"detection recall {_rate_text(result['detection']['recall'])}; "
            f"judge-called {_rate_text(result['gate_coverage']['judge_called_coverage'])}"
        )
    print(f"Report: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--inspect-only", action="store_true", help="validate inputs/dependencies and print the plan (default)")
    actions.add_argument("--run", action="store_true", help="execute off, shadow, and gate")
    actions.add_argument("--report-only", action="store_true", help="regenerate reports from saved per-case results")
    parser.add_argument("--judge", help="required Inspect model identifier for the grader when using --run")
    parser.add_argument("--repetitions", type=int, default=1, help="repeat all three modes (default: 1)")
    parser.add_argument("--out", type=Path, help="output directory; default is results/quality/<UTC timestamp>")
    parser.add_argument("--resume", action="store_true", help="resume a prior --run, verifying saved settings and dataset hashes")
    parser.add_argument("--max-tokens", type=int, default=512, help="grader output token limit (default: 512)")
    parser.add_argument("--timeout", type=int, default=120, help="grader timeout in seconds (default: 120)")
    parser.add_argument("--retries", type=int, default=2, help="grader retries after the initial attempt (default: 2)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    action = "run" if args.resume and not (args.inspect_only or args.run or args.report_only) else (
        "run" if args.run else "report-only" if args.report_only else "inspect-only"
    )
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")
    if args.max_tokens < 1 or args.timeout < 1 or args.retries < 0:
        raise SystemExit("--max-tokens and --timeout must be positive; --retries cannot be negative")
    if args.resume and action != "run":
        raise SystemExit("--resume can only be used with --run")

    if action == "report-only":
        if args.out is None:
            raise SystemExit("--report-only requires --out PATH")
        output_dir = args.out.resolve()
        saved_manifest, rows = _load_report_only(output_dir)
        summary = write_reports(output_dir, saved_manifest, rows)
        write_json(output_dir / "experiment_manifest.json", saved_manifest)
        _print_headlines(summary, output_dir / "report.md")
        return 0

    cases, release_manifest = validate_release()
    if action == "inspect-only":
        dependencies = validate_dependencies()
        print(json.dumps(_inspect_only(cases, dependencies, args.repetitions), indent=2, ensure_ascii=False))
        return 0

    if not args.judge:
        raise SystemExit("--run requires --judge <Inspect model identifier>")
    if args.resume and args.out is None:
        raise SystemExit("--resume requires --out PATH for the existing experiment directory")
    output_dir = args.out.resolve() if args.out else default_output_dir()
    try:
        result = run_experiment(
            cases,
            release_manifest,
            args.judge,
            args.repetitions,
            args.max_tokens,
            args.timeout,
            args.retries,
            output_dir,
            resume=args.resume,
        )
    except Exception as exc:
        print(f"Experiment stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Output directory: {output_dir}", file=sys.stderr)
        return 1
    _print_headlines(result["summary"], output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
