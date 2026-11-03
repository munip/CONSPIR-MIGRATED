"""Lossless v1-to-v2 QA conversion and cached Inspect-log parity helpers."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema
from inspect_ai.log import read_eval_log

from inspect_eval.cb_dataset import (
    as_v2_question_record,
    item_to_sample,
    validate_v2_question_record,
)

PARITY_LABELS = ("hold", "diverge", "abstain")
QA_SCORER_NAME = "conspiracy_judge"
EXPECTED_CORPUS_SHA256 = "22c21ee342796bb1a007be943ddff60ed7d9ece68ed06f381e72411e97fe027b"
EXPECTED_LEGACY_SCHEMA_SHA256 = "3e3a074f70d6c15e53fe9f879055960b860fdefec6aca4aaa81dd0693b57fc70"
EXPECTED_V2_SCHEMA_SHA256 = "2757ade31e8789535dca11b15c79a5c5311222e311c5ac0ca6dca085db9c9e81"
EXPECTED_CORPUS_RECORD_COUNT = 315


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_record_pairs(
    records: list[dict[str, Any]],
    v2_schema: dict,
) -> list[dict[str, Any]]:
    """Validate converted records and compare legacy/v2 evaluation samples."""
    pairs = []
    seen_ids: set[str] = set()
    for record in records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Every legacy QA record must have a nonblank string id")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate legacy QA record id: {record_id}")
        seen_ids.add(record_id)

        converted = as_v2_question_record(record)
        try:
            validate_v2_question_record(converted, v2_schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(
                f"Invalid v2 question_answer record {record_id!r} "
                f"({exc.json_path}): {exc.message}"
            ) from exc
        legacy_sample = item_to_sample(record, record_id)
        v2_sample = item_to_sample(converted, record_id)
        legacy_fields = {
            key: value
            for key, value in converted.items()
            if key not in ("schema_version", "task_type")
        }
        field_mismatches = sorted(
            key for key in set(record) | set(legacy_fields)
            if record.get(key) != legacy_fields.get(key)
        )
        pairs.append({
            "record_id": record_id,
            "legacy": record,
            "v2": converted,
            "legacy_sample": legacy_sample,
            "v2_sample": v2_sample,
            "field_mismatches": field_mismatches,
            "sample_mismatches": {
                "input": legacy_sample.input != v2_sample.input,
                "target": legacy_sample.target != v2_sample.target,
            },
        })
    if not pairs:
        raise ValueError("No legacy QA records were provided")
    return pairs


def _serialize_message(message: Any) -> dict[str, Any]:
    if not hasattr(message, "model_dump"):
        raise TypeError(f"Unsupported Inspect message type: {type(message).__name__}")
    serialized = message.model_dump(mode="json")
    if not isinstance(serialized, dict):
        raise TypeError(f"Inspect message did not serialize to an object: {type(message).__name__}")
    return serialized


def cache_entries_from_eval_log(log: Any) -> list[dict[str, Any]]:
    """Export model messages and existing judge outputs; does not call either model."""
    samples = getattr(log, "samples", None)
    if not samples:
        raise ValueError("Inspect eval log has no sample records")
    entries = []
    seen_keys: set[tuple[str, int]] = set()
    for sample in samples:
        record_id = str(getattr(sample, "id", ""))
        if not record_id:
            raise ValueError("Inspect eval sample has no id")
        epoch_value = getattr(sample, "epoch", None)
        epoch = int(epoch_value) if epoch_value is not None else 1
        key = (record_id, epoch)
        if key in seen_keys:
            raise ValueError(f"Duplicate Inspect sample id/epoch: {key}")
        seen_keys.add(key)

        scores = getattr(sample, "scores", None) or {}
        score = scores.get(QA_SCORER_NAME)
        if score is None:
            matches = [
                candidate for candidate in scores.values()
                if isinstance(getattr(candidate, "metadata", None), dict)
                and "turn_labels" in candidate.metadata
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Inspect sample {record_id!r} has no unambiguous cached "
                    f"{QA_SCORER_NAME} score"
                )
            score = matches[0]
        score_metadata = getattr(score, "metadata", None) or {}
        labels = score_metadata.get("turn_labels")
        if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
            raise ValueError(f"Inspect sample {record_id!r} has no cached turn_labels list")
        messages = [
            _serialize_message(message)
            for message in (getattr(sample, "messages", None) or [])
        ]
        entries.append({
            "record_id": record_id,
            "epoch": epoch,
            "input": getattr(sample, "input", None),
            "target": getattr(sample, "target", None),
            "messages": messages,
            "judge_labels": labels,
            "judge_score_value": getattr(score, "value", None),
            "judge_score_metadata": score_metadata,
            "judge_raw_outputs": getattr(score, "explanation", None),
        })
    return entries


def cache_entries_from_eval_path(path: Path) -> list[dict[str, Any]]:
    return cache_entries_from_eval_log(read_eval_log(path))


def validate_eval_log_settings(
    log: Any,
    *,
    model: str,
    grader: str,
    rubric: str,
) -> dict[str, Any]:
    """Require the supplied model/grader/rubric to match the Inspect log header."""
    eval_spec = getattr(log, "eval", None)
    if eval_spec is None:
        raise ValueError("Inspect eval log has no evaluation header")
    logged_model = getattr(eval_spec, "model", None)
    if logged_model != model:
        raise ValueError(
            f"Model mismatch: log records {logged_model!r}, command supplied {model!r}"
        )

    task_args = getattr(eval_spec, "task_args", None) or {}
    grader_role = task_args.get("grader_role", "grader")
    model_roles = getattr(eval_spec, "model_roles", None) or {}
    grader_config = model_roles.get(grader_role)
    if isinstance(grader_config, list):
        if len(grader_config) != 1:
            raise ValueError(
                f"Expected one logged model for grader role {grader_role!r}"
            )
        grader_config = grader_config[0]
    logged_grader = (
        grader_config.get("model")
        if isinstance(grader_config, dict)
        else getattr(grader_config, "model", None)
    )
    if logged_grader != grader:
        raise ValueError(
            f"Grader mismatch: log records {logged_grader!r}, command supplied {grader!r}"
        )
    logged_rubric = task_args.get("rubric")
    if logged_rubric != rubric:
        raise ValueError(
            f"Rubric mismatch: log records {logged_rubric!r}, command supplied {rubric!r}"
        )

    model_args = getattr(eval_spec, "model_args", None) or {}
    grader_args = (
        grader_config.get("args", {})
        if isinstance(grader_config, dict)
        else getattr(grader_config, "args", {})
    ) or {}
    return {
        "model": logged_model,
        "model_revision": model_args.get("revision"),
        "grader": logged_grader,
        "grader_revision": grader_args.get("revision"),
        "grader_role": grader_role,
        "rubric": logged_rubric,
    }


def _counts_and_rates(entries: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for entry in entries:
        counts.update(entry["judge_labels"])
    total = sum(counts.values())
    answered = counts["hold"] + counts["diverge"]
    return {
        "turn_label_counts": dict(sorted(counts.items())),
        "total_turns": total,
        "divergence_rate_over_answered": counts["diverge"] / answered if answered else None,
        "abstention_rate_over_all_turns": counts["abstain"] / total if total else None,
        "other_label_counts": {
            label: count for label, count in sorted(counts.items())
            if label not in PARITY_LABELS
        },
    }


def build_parity_report(
    pairs: list[dict[str, Any]],
    cache_entries: list[dict[str, Any]],
    *,
    model: str,
    grader: str,
    rubric: str,
) -> dict[str, Any]:
    """Check that the same cached conversations and scores map identically to both schemas."""
    expected_ids = {pair["record_id"] for pair in pairs}
    cache_ids = {entry["record_id"] for entry in cache_entries}
    unknown_ids = sorted(cache_ids - expected_ids)
    missing_ids = sorted(expected_ids - cache_ids)
    mismatched_inputs = []
    mismatched_targets = []
    field_mismatches = {}
    for pair in pairs:
        record_id = pair["record_id"]
        if pair["field_mismatches"]:
            field_mismatches[record_id] = pair["field_mismatches"]
        if pair["sample_mismatches"]["input"]:
            mismatched_inputs.append(record_id)
        if pair["sample_mismatches"]["target"]:
            mismatched_targets.append(record_id)

    pair_by_id = {pair["record_id"]: pair for pair in pairs}
    cache_input_mismatches = []
    cache_target_mismatches = []
    for entry in cache_entries:
        pair = pair_by_id.get(entry["record_id"])
        if pair is None:
            continue
        if (
            entry["input"] != pair["legacy_sample"].input
            or entry["input"] != pair["v2_sample"].input
        ):
            cache_input_mismatches.append(entry["record_id"])
        if (
            entry["target"] != pair["legacy_sample"].target
            or entry["target"] != pair["v2_sample"].target
        ):
            cache_target_mismatches.append(entry["record_id"])

    duplicate_keys = []
    keys: set[tuple[str, int]] = set()
    for entry in cache_entries:
        key = (entry["record_id"], entry["epoch"])
        if key in keys:
            duplicate_keys.append({"record_id": key[0], "epoch": key[1]})
        keys.add(key)

    legacy_replay = []
    v2_replay = []
    for entry in cache_entries:
        pair = pair_by_id.get(entry["record_id"])
        if pair is None:
            continue
        replayed_score = {
            "record_id": entry["record_id"],
            "epoch": entry["epoch"],
            "messages": entry["messages"],
            "judge_labels": entry["judge_labels"],
            "judge_score_value": entry["judge_score_value"],
        }
        legacy_replay.append({
            **replayed_score,
            "input": pair["legacy_sample"].input,
            "target": pair["legacy_sample"].target,
        })
        v2_replay.append({
            **replayed_score,
            "input": pair["v2_sample"].input,
            "target": pair["v2_sample"].target,
        })
    legacy_counts = _counts_and_rates(legacy_replay)
    v2_counts = _counts_and_rates(v2_replay)
    score_mismatch_count = sum(
        legacy["messages"] != v2["messages"]
        or legacy["judge_labels"] != v2["judge_labels"]
        or legacy["judge_score_value"] != v2["judge_score_value"]
        for legacy, v2 in zip(legacy_replay, v2_replay)
    )
    exact = (
        len(pairs) == len(expected_ids)
        and not unknown_ids
        and not missing_ids
        and not duplicate_keys
        and not field_mismatches
        and not mismatched_inputs
        and not mismatched_targets
        and not cache_input_mismatches
        and not cache_target_mismatches
        and score_mismatch_count == 0
        and legacy_replay == v2_replay
        and legacy_counts == v2_counts
    )
    return {
        "parity_pass": exact,
        "settings": {"model": model, "grader": grader, "rubric": rubric},
        "record_count": len(pairs),
        "cached_eval_sample_count": len(cache_entries),
        "cached_unique_record_count": len(cache_ids),
        "missing_record_ids": missing_ids,
        "unknown_record_ids": unknown_ids,
        "duplicate_record_epoch_keys": duplicate_keys,
        "legacy_v2_field_mismatches": field_mismatches,
        "legacy_v2_input_mismatch_ids": mismatched_inputs,
        "legacy_v2_target_mismatch_ids": mismatched_targets,
        "cache_input_mismatch_ids": sorted(set(cache_input_mismatches)),
        "cache_target_mismatch_ids": sorted(set(cache_target_mismatches)),
        "legacy": legacy_counts,
        "v2": v2_counts,
        "per_item_score_mismatch_count": score_mismatch_count,
        "aggregate_score_mismatch": legacy_counts != v2_counts,
        "replay_method": (
            "The same logged model messages and cached judge labels were mapped "
            "to both representations; no model or judge calls were made."
        ),
    }


def build_conversion_manifest(
    pairs: list[dict[str, Any]],
    *,
    corpus_sha256: str,
    legacy_schema_sha256: str,
    v2_schema_sha256: str,
) -> dict[str, Any]:
    return {
        "corpus_sha256": corpus_sha256,
        "legacy_schema_sha256": legacy_schema_sha256,
        "v2_schema_sha256": v2_schema_sha256,
        "record_count": len(pairs),
        "converted_records": [
            {
                "id": pair["record_id"],
                "added_fields": {
                    "schema_version": pair["v2"]["schema_version"],
                    "task_type": pair["v2"]["task_type"],
                },
                "preserved_fields": sorted(pair["legacy"]),
                "field_mismatches": pair["field_mismatches"],
                "input_equal": not pair["sample_mismatches"]["input"],
                "target_equal": not pair["sample_mismatches"]["target"],
            }
            for pair in pairs
        ],
    }
