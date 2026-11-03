"""Single-label dominant-trait pilot for ConspirED."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import generate, system_message

TRAITS = (
    "contradictory",
    "overriding_suspicion",
    "nefarious_intent",
    "persecuted_victim",
    "immune_to_evidence",
    "reinterpreting_randomness",
)

TRAIT_DEFINITIONS = {
    "contradictory": "Contradictory claims or internal inconsistencies",
    "overriding_suspicion": "Overriding suspicion of authority or institutions",
    "nefarious_intent": "Attribution of nefarious intent",
    "persecuted_victim": "Self or group portrayed as a persecuted victim",
    "immune_to_evidence": "Claims immune to contrary evidence",
    "reinterpreting_randomness": "Re-interpreting random events as meaningful",
}

ARMS = {
    "snippet_only": False,
    "context1000": True,
}

EXPECTED_MODEL_ID = "hf/Qwen/Qwen2.5-3B-Instruct"
EXPECTED_MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EXPECTED_SOURCE_SHA256 = {
    "train": "285fdf35b507c8d4306661cc00f1f987da018f237e4c332724935231f8981dfb",
    "test": "a4b52c52eee870ff18ab21d2f9f421c69bd09f6e816a5179d5272e71c0e5fa8a",
}

_SYSTEM_PROMPT = """You identify the single most salient CONSPIR trait exhibited by the annotated snippet.
Use supplied context only to interpret the annotated snippet; classify the snippet, not the whole source document.

Choose exactly one trait from this list:
""" + "\n".join(
    f'- {trait}: {TRAIT_DEFINITIONS[trait]}' for trait in TRAITS
) + """

Return only one trait ID exactly as written, for example nefarious_intent.
Do not return quotes, an array, multiple traits, or an explanation.
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_import_manifest(
    dataset_path: Path,
    manifest_path: Path | None = None,
    split: str = "train",
) -> dict[str, Any]:
    """Verify that the imported dataset uses the expected source and context column."""
    if split not in ("train", "test"):
        raise ValueError("ConspirED split must be 'train' or 'test'")
    dataset_path = Path(dataset_path)
    manifest_path = manifest_path or dataset_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("context_column_used") != "context1000":
        raise ValueError("ConspirED import manifest must set context_column_used to context1000")
    if manifest.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("ConspirED source XLSX hashes do not match the pinned inputs")
    expected_output = (manifest.get("output_sha256") or {}).get(split)
    actual_output = sha256_file(dataset_path)
    if expected_output != actual_output:
        raise ValueError(
            f"ConspirED {split} JSONL hash does not match the import manifest: "
            f"expected {expected_output!r}, got {actual_output}"
        )
    if manifest.get("train_test_document_overlap") != 0:
        raise ValueError("ConspirED import manifest reports train/test document overlap")
    return manifest


def load_pilot_data(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Load valid training rows and exclude rows without the context needed by both arms."""
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
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
            if source.get("upstream_split") != "train":
                raise ValueError(f"Record {sample_id!r} is not from the training split")
            if row.get("dominant_trait") not in TRAITS:
                raise ValueError(f"Record {sample_id!r} has an invalid dominant_trait")
            if not isinstance(metadata, dict):
                raise ValueError(f"Record {sample_id!r} has invalid metadata")
            context = metadata.get("context1000")
            if context is not None and not isinstance(context, str):
                raise ValueError(f"Record {sample_id!r} has a non-string context1000")
            if not isinstance(context, str) or not context.strip():
                excluded.append({
                    "id": sample_id,
                    "document_id": source["document_id"],
                    "reason": "missing_context1000",
                })
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No eligible training rows found in {path}")
    return rows, excluded


def select_balanced_sample(
    rows: list[dict[str, Any]],
    items_per_class: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Choose a deterministic balanced sample, favoring unused source documents."""
    if items_per_class < 1:
        raise ValueError("items_per_class must be positive")
    rng = random.Random(seed)
    by_trait: dict[str, list[dict[str, Any]]] = {trait: [] for trait in TRAITS}
    for row in rows:
        trait = row.get("dominant_trait")
        if trait not in by_trait:
            raise ValueError(f"Invalid dominant_trait on sample {row.get('id')!r}")
        by_trait[trait].append(row)
    for trait, candidates in by_trait.items():
        if len(candidates) < items_per_class:
            raise ValueError(
                f"Need {items_per_class} samples for {trait}; found {len(candidates)}"
            )
        rng.shuffle(candidates)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    document_counts: Counter[str] = Counter()
    for _ in range(items_per_class):
        for trait in TRAITS:
            candidates = [
                row for row in by_trait[trait] if row["id"] not in selected_ids
            ]
            chosen = min(
                candidates,
                key=lambda row: document_counts[row["source"]["document_id"]],
            )
            selected.append(chosen)
            selected_ids.add(chosen["id"])
            document_counts[chosen["source"]["document_id"]] += 1
    return selected


def majority_trait(rows: list[dict[str, Any]]) -> str:
    """Return the most common label, breaking ties in canonical trait order."""
    if not rows:
        raise ValueError("Cannot determine the majority trait from an empty development set")
    counts = Counter(row["dominant_trait"] for row in rows)
    return max(TRAITS, key=lambda trait: (counts[trait], -TRAITS.index(trait)))


def parse_dominant_prediction(completion: str) -> str:
    """Parse one canonical trait ID, optionally wrapped as a JSON string."""
    text = completion.strip()
    if text in TRAITS:
        return text
    try:
        prediction = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Output is not one canonical dominant-trait ID") from exc
    if not isinstance(prediction, str) or prediction not in TRAITS:
        raise ValueError("Output must be one canonical dominant-trait ID")
    return prediction


def build_sample_input(row: dict[str, Any], include_context: bool) -> str:
    if include_context:
        return (
            f"Context (context1000):\n{row['metadata']['context1000']}\n\n"
            f"Annotated snippet:\n{row['text']}"
        )
    return f"Annotated snippet:\n{row['text']}"


@scorer(metrics=[])
def dominant_trait_pilot_scorer():
    async def score(state, target: Target) -> Score:
        item = state.metadata["item"]
        expected = json.loads(target.text)
        if expected not in TRAITS:
            raise ValueError(f"Invalid target dominant_trait for {item.get('id')!r}")
        completion = state.output.completion
        try:
            predicted: str | None = parse_dominant_prediction(completion)
            invalid = False
        except ValueError:
            predicted = None
            invalid = True
        result = {
            "sample_id": str(getattr(state, "sample_id", None) or item["id"]),
            "document_id": item["source"]["document_id"],
            "predicted": predicted,
            "target": expected,
            "invalid": invalid,
        }
        return Score(
            value=json.dumps(result, ensure_ascii=False),
            explanation=completion,
        )

    return score


def build_arm_task(
    rows: list[dict[str, Any]],
    arm: str,
) -> Task:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {tuple(ARMS)}")
    samples = [
        Sample(
            input=build_sample_input(row, include_context=ARMS[arm]),
            target=json.dumps(row["dominant_trait"]),
            id=row["id"],
            metadata={"item": row},
        )
        for row in rows
    ]
    return Task(
        dataset=MemoryDataset(samples, name=f"conspired-dominant-{arm}"),
        solver=[system_message(_SYSTEM_PROMPT), generate()],
        scorer=dominant_trait_pilot_scorer(),
        name=f"conspired_dominant_trait_{arm}",
    )


def rows_from_eval_log(log: Any) -> list[dict[str, Any]]:
    samples = getattr(log, "samples", None)
    if not samples:
        raise ValueError("Inspect eval log has no sample records")
    rows = []
    for sample in samples:
        scores = getattr(sample, "scores", None) or {}
        score = scores.get("dominant_trait_pilot_scorer")
        if score is None:
            raise ValueError(
                f"Eval sample {getattr(sample, 'id', '<unknown>')!r} "
                "has no dominant_trait_pilot_scorer score"
            )
        try:
            row = json.loads(str(score.value))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid pilot score for sample {getattr(sample, 'id', '<unknown>')!r}"
            ) from exc
        required = {"sample_id", "document_id", "predicted", "target", "invalid"}
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(
                f"Malformed dominant-trait score for sample {getattr(sample, 'id', '<unknown>')!r}"
            )
        if row["sample_id"] != str(getattr(sample, "id", "")):
            raise ValueError(
                f"Score sample ID {row['sample_id']!r} does not match "
                f"Inspect sample {getattr(sample, 'id', '<unknown>')!r}"
            )
        completion = getattr(getattr(sample, "output", None), "completion", None)
        if not isinstance(completion, str):
            completion = getattr(score, "explanation", None)
        if not isinstance(completion, str):
            raise ValueError(
                f"Eval sample {getattr(sample, 'id', '<unknown>')!r} "
                "has no raw model completion to parse"
            )
        try:
            row["predicted"] = parse_dominant_prediction(completion)
            row["invalid"] = False
        except ValueError:
            row["predicted"] = None
            row["invalid"] = True
        rows.append(row)
    return rows


def _validate_scored_rows(rows: list[dict[str, Any]]) -> None:
    sample_ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise ValueError("Scored rows must contain nonblank sample IDs")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Scored rows contain duplicate sample IDs")
    for row in rows:
        if not isinstance(row.get("document_id"), str) or not row["document_id"]:
            raise ValueError(f"Scored row {row.get('sample_id')!r} has no document ID")
        if row.get("target") not in TRAITS:
            raise ValueError(f"Scored row {row['sample_id']!r} has an invalid target")
        if row.get("predicted") is not None and row["predicted"] not in TRAITS:
            raise ValueError(f"Scored row {row['sample_id']!r} has an invalid prediction")
        if not isinstance(row.get("invalid"), bool):
            raise ValueError(f"Scored row {row['sample_id']!r} has invalid status")
        if row["invalid"] != (row["predicted"] is None):
            raise ValueError(f"Scored row {row['sample_id']!r} has inconsistent invalid status")


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _validate_scored_rows(rows)
    if not rows:
        raise ValueError("Cannot summarize an empty pilot arm")
    matrix = {actual: {predicted: 0 for predicted in TRAITS} for actual in TRAITS}
    per_trait = {}
    correct_total = 0
    for row in rows:
        target = row["target"]
        predicted = row["predicted"]
        correct = predicted == target
        correct_total += int(correct)
        if predicted is not None:
            matrix[target][predicted] += 1
    for trait in TRAITS:
        class_rows = [row for row in rows if row["target"] == trait]
        correct = sum(row["predicted"] == trait for row in class_rows)
        per_trait[trait] = {
            "support": len(class_rows),
            "correct": correct,
            "incorrect": len(class_rows) - correct,
            "recall": correct / len(class_rows) if class_rows else None,
        }
    valid_count = sum(not row["invalid"] for row in rows)
    return {
        "sample_count": len(rows),
        "valid_single_label_count": valid_count,
        "invalid_output_count": len(rows) - valid_count,
        "valid_single_label_rate": valid_count / len(rows),
        "exact_accuracy": correct_total / len(rows),
        "correct_count": correct_total,
        "confusion_matrix": matrix,
        "per_trait": per_trait,
    }


def build_pilot_report(
    arm_rows: dict[str, list[dict[str, Any]]],
    sample_ids: list[str],
    baseline_trait: str,
) -> dict[str, Any]:
    if set(arm_rows) != set(ARMS):
        raise ValueError(f"Expected pilot arms {tuple(ARMS)}")
    if baseline_trait not in TRAITS:
        raise ValueError(f"Invalid baseline trait {baseline_trait!r}")
    rows_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, rows in arm_rows.items():
        _validate_scored_rows(rows)
        by_id = {row["sample_id"]: row for row in rows}
        if set(by_id) != set(sample_ids):
            raise ValueError(f"Arm {arm!r} did not return exactly the planned sample IDs")
        rows_by_arm[arm] = by_id

    snippet = rows_by_arm["snippet_only"]
    context = rows_by_arm["context1000"]
    baseline_rows = [
        {
            **row,
            "predicted": baseline_trait,
            "invalid": False,
        }
        for row in snippet.values()
    ]
    paired = {
        "both_correct": 0,
        "snippet_only_correct_context_wrong": 0,
        "context_correct_snippet_wrong": 0,
        "both_wrong": 0,
        "prediction_changed": 0,
    }
    disagreements = []
    for sample_id in sample_ids:
        snippet_row = snippet[sample_id]
        context_row = context[sample_id]
        snippet_correct = snippet_row["predicted"] == snippet_row["target"]
        context_correct = context_row["predicted"] == context_row["target"]
        if snippet_correct and context_correct:
            paired["both_correct"] += 1
        elif snippet_correct:
            paired["snippet_only_correct_context_wrong"] += 1
        elif context_correct:
            paired["context_correct_snippet_wrong"] += 1
        else:
            paired["both_wrong"] += 1
        if snippet_row["predicted"] != context_row["predicted"]:
            paired["prediction_changed"] += 1
            disagreements.append({
                "sample_id": sample_id,
                "document_id": snippet_row["document_id"],
                "target": snippet_row["target"],
                "snippet_only_prediction": snippet_row["predicted"],
                "context1000_prediction": context_row["predicted"],
            })
    paired["accuracy_difference_context_minus_snippet"] = (
        metric_summary(list(context.values()))["exact_accuracy"]
        - metric_summary(list(snippet.values()))["exact_accuracy"]
    )
    return {
        "evaluation_status": "diagnostic_pilot_only",
        "arms": {
            arm: metric_summary(rows)
            for arm, rows in arm_rows.items()
        },
        "majority_baseline": {
            "predicted_trait": baseline_trait,
            **metric_summary(baseline_rows),
        },
        "context_comparison": {
            "paired_outcomes": paired,
            "disagreements": disagreements,
        },
        "performance_claim": None,
    }
