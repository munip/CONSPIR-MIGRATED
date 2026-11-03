"""Full held-out ConspirED dominant-trait evaluation and document bootstrap."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .dominant_trait_pilot import (
    ARMS,
    EXPECTED_SOURCE_SHA256,
    TRAITS,
    metric_summary,
    verify_import_manifest,
)

DEFAULT_BOOTSTRAP_REPETITIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 42


def load_test_data(path: Path) -> list[dict[str, Any]]:
    """Load every official test row; missing labels or context are errors."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            sample_id = row.get("id")
            source = row.get("source")
            metadata = row.get("metadata")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"{path}:{line_number}: missing nonblank id")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate ConspirED sample id: {sample_id}")
            seen_ids.add(sample_id)
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"Record {sample_id!r} has blank text")
            if not isinstance(source, dict) or not isinstance(
                source.get("document_id"), str
            ) or not source["document_id"].strip():
                raise ValueError(f"Record {sample_id!r} has no source.document_id")
            if source.get("upstream_split") != "test":
                raise ValueError(f"Record {sample_id!r} is not from the test split")
            if row.get("dominant_trait") not in TRAITS:
                raise ValueError(f"Record {sample_id!r} has an invalid dominant_trait")
            if not isinstance(metadata, dict):
                raise ValueError(f"Record {sample_id!r} has invalid metadata")
            context = metadata.get("context1000")
            if not isinstance(context, str) or not context.strip():
                raise ValueError(
                    f"Test record {sample_id!r} has missing or blank context1000"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No test rows found in {path}")
    return rows


def verify_test_dataset(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify the pinned test import and require complete, unfiltered coverage."""
    manifest = verify_import_manifest(path, split="test")
    if manifest.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("ConspirED source XLSX hashes do not match the pinned inputs")
    rows = load_test_data(path)
    if len(rows) != manifest.get("test_records"):
        raise ValueError(
            "Loaded test row count does not match the import manifest: "
            f"expected {manifest.get('test_records')!r}, got {len(rows)}"
        )
    if manifest.get("train_test_document_overlap") != 0:
        raise ValueError("ConspirED import manifest reports train/test document overlap")
    return manifest, rows


def training_majority_baseline(
    train_path: Path,
) -> tuple[str, int, int]:
    """Choose a training-only baseline outside the deterministic balanced pilot."""
    from .dominant_trait_pilot import load_pilot_data, select_balanced_sample

    train_manifest = verify_import_manifest(train_path, split="train")
    rows, excluded = load_pilot_data(train_path)
    pilot_ids = {
        row["id"]
        for row in select_balanced_sample(rows, items_per_class=10, seed=42)
    }
    development_rows = [
        row for row in rows if row["id"] not in pilot_ids
    ]
    counts = Counter(row["dominant_trait"] for row in development_rows)
    baseline = max(TRAITS, key=lambda trait: (counts[trait], -TRAITS.index(trait)))
    if len(rows) + len(excluded) != train_manifest.get("train_records"):
        raise ValueError("Training rows and exclusions do not match the import manifest")
    return baseline, counts[baseline], len(development_rows)


def _validate_paired_rows(
    arm_rows: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], list[str]]:
    if set(arm_rows) != set(ARMS):
        raise ValueError(f"Expected pilot arms {tuple(ARMS)}")
    for rows in arm_rows.values():
        metric_summary(rows)
    id_sets = [{row["sample_id"] for row in rows} for rows in arm_rows.values()]
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise ValueError("Evaluation arms do not contain the same sample IDs")
    by_arm = {
        arm: {row["sample_id"]: row for row in rows}
        for arm, rows in arm_rows.items()
    }
    reference = by_arm["snippet_only"]
    for arm in ARMS:
        if {
            sample_id: (row["document_id"], row["target"])
            for sample_id, row in by_arm[arm].items()
        } != {
            sample_id: (row["document_id"], row["target"])
            for sample_id, row in reference.items()
        }:
            raise ValueError(f"Arm {arm!r} does not preserve paired targets and documents")
    sample_ids = [row["sample_id"] for row in arm_rows["snippet_only"]]
    documents = sorted({row["document_id"] for row in arm_rows["snippet_only"]})
    return sample_ids, documents


def _single_label_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot score an empty test sample")
    matrix = {
        actual: {predicted: 0 for predicted in (*TRAITS, "invalid_output")}
        for actual in TRAITS
    }
    per_trait: dict[str, dict[str, Any]] = {}
    for trait in TRAITS:
        support = sum(row["target"] == trait for row in rows)
        true_positive = sum(
            row["target"] == trait and row["predicted"] == trait for row in rows
        )
        predicted_count = sum(row["predicted"] == trait for row in rows)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        per_trait[trait] = {
            "support": support,
            "predicted_count": predicted_count,
            "correct": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    for row in rows:
        output = row["predicted"] if row["predicted"] in TRAITS else "invalid_output"
        matrix[row["target"]][output] += 1
    correct = sum(row["predicted"] == row["target"] for row in rows)
    valid = sum(not row["invalid"] for row in rows)
    return {
        "sample_count": len(rows),
        "valid_single_label_count": valid,
        "invalid_output_count": len(rows) - valid,
        "valid_single_label_rate": valid / len(rows),
        "exact_accuracy": correct / len(rows),
        "correct_count": correct,
        "macro_f1": sum(row["f1"] for row in per_trait.values()) / len(TRAITS),
        "confusion_matrix": matrix,
        "per_trait": per_trait,
        "f1_zero_division": 0,
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = lower if position.is_integer() else lower + 1
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_document_bootstrap(
    arm_rows: dict[str, list[dict[str, Any]]],
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap paired macro-F1 and accuracy differences by source document."""
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    _, documents = _validate_paired_rows(arm_rows)
    grouped = {
        arm: {
            document: [row for row in rows if row["document_id"] == document]
            for document in documents
        }
        for arm, rows in arm_rows.items()
    }
    rng = random.Random(seed)
    bootstrap_values = {
        metric: {"context1000": [], "snippet_only": [], "difference": []}
        for metric in ("macro_f1", "exact_accuracy")
    }
    for _ in range(repetitions):
        sampled_documents = [rng.choice(documents) for _ in documents]
        sampled = {
            arm: [
                row
                for document in sampled_documents
                for row in grouped[arm][document]
            ]
            for arm in ARMS
        }
        summaries = {
            arm: _single_label_metrics(rows)
            for arm, rows in sampled.items()
        }
        for metric in bootstrap_values:
            context_value = summaries["context1000"][metric]
            snippet_value = summaries["snippet_only"][metric]
            bootstrap_values[metric]["context1000"].append(context_value)
            bootstrap_values[metric]["snippet_only"].append(snippet_value)
            bootstrap_values[metric]["difference"].append(
                context_value - snippet_value
            )

    def interval(values: list[float]) -> dict[str, float]:
        return {
            "lower": _quantile(values, 0.025),
            "upper": _quantile(values, 0.975),
        }

    return {
        "resampling_unit": "source.document_id",
        "document_count": len(documents),
        "repetitions": repetitions,
        "seed": seed,
        "interval_method": "paired percentile bootstrap with linear interpolation",
        "metric_note": (
            "Macro-F1 always averages the same six traits; precision, recall, "
            "and F1 are zero when their denominator is zero in a bootstrap draw."
        ),
        "macro_f1": {
            "snippet_only": interval(bootstrap_values["macro_f1"]["snippet_only"]),
            "context1000": interval(bootstrap_values["macro_f1"]["context1000"]),
            "context1000_minus_snippet_only": interval(
                bootstrap_values["macro_f1"]["difference"]
            ),
        },
        "exact_accuracy": {
            "snippet_only": interval(
                bootstrap_values["exact_accuracy"]["snippet_only"]
            ),
            "context1000": interval(
                bootstrap_values["exact_accuracy"]["context1000"]
            ),
            "context1000_minus_snippet_only": interval(
                bootstrap_values["exact_accuracy"]["difference"]
            ),
        },
    }


def build_test_report(
    arm_rows: dict[str, list[dict[str, Any]]],
    test_rows: list[dict[str, Any]],
    baseline_trait: str,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if baseline_trait not in TRAITS:
        raise ValueError(f"Invalid baseline trait {baseline_trait!r}")
    expected_ids = [row["id"] for row in test_rows]
    actual_ids, documents = _validate_paired_rows(arm_rows)
    if len(expected_ids) != len(set(expected_ids)) or set(expected_ids) != set(actual_ids):
        raise ValueError("Evaluation arms do not contain exactly all official test IDs")
    expected = {
        row["id"]: (row["source"]["document_id"], row["dominant_trait"])
        for row in test_rows
    }
    for arm, rows in arm_rows.items():
        actual = {
            row["sample_id"]: (row["document_id"], row["target"])
            for row in rows
        }
        if actual != expected:
            raise ValueError(f"Arm {arm!r} predictions do not match test metadata")
    baseline_rows = [
        {
            "sample_id": row["id"],
            "document_id": row["source"]["document_id"],
            "predicted": baseline_trait,
            "target": row["dominant_trait"],
            "invalid": False,
        }
        for row in test_rows
    ]
    summaries = {
        arm: _single_label_metrics(rows)
        for arm, rows in arm_rows.items()
    }
    return {
        "evaluation_status": "descriptive_held_out_test",
        "primary_metric": "macro_f1",
        "arms": summaries,
        "majority_baseline": {
            "predicted_trait": baseline_trait,
            **_single_label_metrics(baseline_rows),
        },
        "test_class_support": {
            trait: sum(row["dominant_trait"] == trait for row in test_rows)
            for trait in TRAITS
        },
        "rare_class_caveat": (
            "Only three official test examples have the contradictory label; "
            "class-specific estimates for it are highly uncertain."
        ),
        "test_sample_count": len(test_rows),
        "source_document_count": len(documents),
        "context_comparison": {
            "macro_f1_difference_context1000_minus_snippet_only": (
                summaries["context1000"]["macro_f1"]
                - summaries["snippet_only"]["macro_f1"]
            ),
            "exact_accuracy_difference_context1000_minus_snippet_only": (
                summaries["context1000"]["exact_accuracy"]
                - summaries["snippet_only"]["exact_accuracy"]
            ),
            "paired_document_bootstrap": paired_document_bootstrap(
                arm_rows, repetitions=repetitions, seed=seed
            ),
        },
        "items": [
            {
                "sample_id": row["id"],
                "document_id": row["source"]["document_id"],
                "target": row["dominant_trait"],
                **{
                    f"{arm}_prediction": next(
                        scored["predicted"]
                        for scored in arm_rows[arm]
                        if scored["sample_id"] == row["id"]
                    )
                    for arm in ARMS
                },
                **{
                    f"{arm}_invalid": next(
                        scored["invalid"]
                        for scored in arm_rows[arm]
                        if scored["sample_id"] == row["id"]
                    )
                    for arm in ARMS
                },
            }
            for row in test_rows
        ],
        "performance_claim": None,
    }
