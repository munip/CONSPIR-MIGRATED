"""Grouped 2x2 prompt/context experiment for the six labeled ConspirED traits."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import generate, system_message

from .trait_classification import parse_trait_prediction

SCORED_TRAITS = (
    "contradictory",
    "overriding_suspicion",
    "nefarious_intent",
    "persecuted_victim",
    "immune_to_evidence",
    "reinterpreting_randomness",
)

ARMS = {
    "snippet_only": (False, False),
    "context_only": (True, False),
    "demos_only": (False, True),
    "context_plus_demos": (True, True),
}

EXPECTED_DATASET_SHA256 = "149f80d46018fef0c7ffc0b3be0d9cc6ee33d146b9a7edb9d87d6eb3a7310f45"
DEFAULT_BOOTSTRAP_REPETITIONS = 1000
IMPROVEMENT_THRESHOLD = 0.05

_TRAIT_DEFINITIONS = {
    "contradictory": "Contradictory claims or internal inconsistencies",
    "overriding_suspicion": "Overriding suspicion of authority or institutions",
    "nefarious_intent": "Attribution of nefarious intent",
    "persecuted_victim": "Self or group portrayed as a persecuted victim",
    "immune_to_evidence": "Claims immune to contrary evidence",
    "reinterpreting_randomness": "Re-interpreting random events as meaningful",
}

_BASE_PROMPT = """You identify CONSPIR traits in an annotated text snippet.
Classify only the snippet, using context only to interpret it when context is supplied.
A snippet may exhibit zero or more traits. Include a trait only when clearly exhibited.

The six scored traits are:
""" + "\n".join(
    f"- {trait}: {_TRAIT_DEFINITIONS[trait]}" for trait in SCORED_TRAITS
) + """

Return only a JSON array containing trait IDs from this six-trait list.
For example: ["nefarious_intent", "persecuted_victim"]
If no traits are present, return [].
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def load_experiment_data(
    path: Path,
    expected_sha256: str | None = EXPECTED_DATASET_SHA256,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Load fully labeled rows with context1000 and report rows excluded for missing context."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"Dataset SHA-256 mismatch for {path}: expected {expected_sha256}, got {digest}"
        )

    all_rows = _read_jsonl(path)
    seen_ids: set[str] = set()
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for row in all_rows:
        record_id = row.get("id")
        text = row.get("text")
        source = row.get("source")
        document_id = source.get("document_id") if isinstance(source, dict) else None
        annotations = row.get("trait_annotations")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("Every record must have a nonblank string id")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate record id: {record_id}")
        seen_ids.add(record_id)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Record {record_id!r} has empty text")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"Record {record_id!r} has no source.document_id")
        if not isinstance(annotations, dict):
            raise ValueError(f"Record {record_id!r} has invalid trait_annotations")
        invalid_traits = [
            trait for trait in SCORED_TRAITS
            if annotations.get(trait) not in ("present", "absent")
        ]
        if invalid_traits:
            raise ValueError(
                f"Record {record_id!r} lacks known labels for: {', '.join(invalid_traits)}"
            )

        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Record {record_id!r} has invalid metadata")
        context = metadata.get("context1000")
        if context is not None and not isinstance(context, str):
            raise ValueError(f"Record {record_id!r} has a non-string context1000")
        if context is None or not context.strip():
            excluded.append({
                "id": record_id,
                "document_id": document_id,
                "reason": "missing_context1000",
            })
            continue
        eligible.append(row)

    if not eligible:
        raise ValueError("No rows have nonblank context1000")
    for trait in SCORED_TRAITS:
        if not any(row["trait_annotations"][trait] == "present" for row in eligible):
            raise ValueError(f"No eligible positive examples for trait {trait!r}")
    return eligible, excluded


def _document_id(row: dict[str, Any]) -> str:
    return str(row["source"]["document_id"])


def _select_demonstrations(
    rows: list[dict[str, Any]], validation_documents: set[str]
) -> list[dict[str, Any]]:
    training = [row for row in rows if _document_id(row) not in validation_documents]
    demonstrations: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_documents: set[str] = set()

    def choose(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidates:
            raise ValueError("A training fold has no positive example for a scored trait")
        unused = [row for row in candidates if row["id"] not in selected_ids]
        if not unused:
            return candidates[0]
        return next(
            (row for row in unused if _document_id(row) not in selected_documents),
            unused[0],
        )

    for trait in SCORED_TRAITS:
        chosen = choose([
            row for row in training
            if row["trait_annotations"][trait] == "present"
        ])
        selected_ids.add(chosen["id"])
        selected_documents.add(_document_id(chosen))
        demonstrations.append({
            "id": chosen["id"],
            "document_id": _document_id(chosen),
            "text": chosen["text"],
            "traits": [
                candidate for candidate in SCORED_TRAITS
                if chosen["trait_annotations"][candidate] == "present"
            ],
        })

    negative_examples = [
        row for row in training
        if all(row["trait_annotations"][trait] == "absent" for trait in SCORED_TRAITS)
    ]
    if negative_examples:
        chosen = choose(negative_examples)
        demonstrations.append({
            "id": chosen["id"],
            "document_id": _document_id(chosen),
            "text": chosen["text"],
            "traits": [],
        })
    return demonstrations


def build_fold_plan(
    rows: list[dict[str, Any]], folds: int = 5, seed: int = 42
) -> dict[str, Any]:
    """Assign whole documents to deterministic folds and select training-only examples."""
    if folds < 2:
        raise ValueError("At least two folds are required")
    sample_ids = [row.get("id") for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Fold planning requires unique sample IDs")
    documents = sorted({_document_id(row) for row in rows})
    if len(documents) < folds:
        raise ValueError(f"Need at least {folds} document groups, found {len(documents)}")

    group_sizes = {document: 0 for document in documents}
    for row in rows:
        group_sizes[_document_id(row)] += 1
    tie_order = documents[:]
    random.Random(seed).shuffle(tie_order)
    tie_rank = {document: index for index, document in enumerate(tie_order)}
    ordered_documents = sorted(
        documents,
        key=lambda document: (-group_sizes[document], tie_rank[document]),
    )
    fold_documents: list[list[str]] = [[] for _ in range(folds)]
    fold_sample_counts = [0] * folds
    for document in ordered_documents:
        fold_index = min(
            range(folds),
            key=lambda index: (fold_sample_counts[index], len(fold_documents[index]), index),
        )
        fold_documents[fold_index].append(document)
        fold_sample_counts[fold_index] += group_sizes[document]
    document_fold = {
        document: fold_index
        for fold_index, document_ids in enumerate(fold_documents)
        for document in document_ids
    }

    fold_plans = []
    for fold_index, validation_documents in enumerate(fold_documents):
        validation_set = set(validation_documents)
        validation_rows = [row for row in rows if _document_id(row) in validation_set]
        demonstrations = _select_demonstrations(rows, validation_set)
        fold_plans.append({
            "fold": fold_index + 1,
            "validation_document_ids": sorted(validation_set),
            "training_document_ids": sorted(set(documents) - validation_set),
            "validation_sample_ids": [row["id"] for row in validation_rows],
            "training_sample_count": len(rows) - len(validation_rows),
            "demonstrations": [
                {
                    "id": demo["id"],
                    "document_id": demo["document_id"],
                    "traits": demo["traits"],
                }
                for demo in demonstrations
            ],
        })
    return {
        "fold_count": folds,
        "seed": seed,
        "document_count": len(documents),
        "sample_count": len(rows),
        "document_to_fold": {
            document: document_fold[document] + 1 for document in sorted(document_fold)
        },
        "folds": fold_plans,
    }


def build_system_prompt(demonstrations: list[dict[str, Any]] | None = None) -> str:
    if not demonstrations:
        return _BASE_PROMPT
    examples = [
        json.dumps(
            {"text": demo["text"], "traits": demo["traits"]},
            ensure_ascii=False,
        )
        for demo in demonstrations
    ]
    return (
        _BASE_PROMPT
        + "\nLabeled training examples:\n"
        + "\n".join(examples)
        + "\n"
    )


def build_sample_input(row: dict[str, Any], include_context: bool) -> str:
    text = row["text"]
    if not include_context:
        return f"Annotated text:\n{text}"
    metadata = row.get("metadata")
    context = metadata.get("context1000") if isinstance(metadata, dict) else None
    if not isinstance(context, str) or not context.strip():
        raise ValueError(f"Record {row.get('id')!r} has no nonblank metadata.context1000")
    return f"Context (context1000):\n{context}\n\nAnnotated text:\n{text}"


def _parse_six_trait_prediction(completion: str) -> list[str]:
    prediction = parse_trait_prediction(completion)
    unsupported = sorted(set(prediction) - set(SCORED_TRAITS))
    if unsupported:
        raise ValueError(f"Output contains traits outside this experiment: {unsupported}")
    return prediction


@scorer(metrics=[])
def trait_prompt_experiment_scorer():
    async def score(state, target: Target) -> Score:
        item = state.metadata["item"]
        source = item["source"]
        target_traits = json.loads(target.text)
        if (
            not isinstance(target_traits, list)
            or any(trait not in SCORED_TRAITS for trait in target_traits)
        ):
            raise ValueError(f"Invalid target labels for {item.get('id')!r}")
        completion = state.output.completion
        try:
            predicted = _parse_six_trait_prediction(completion)
            invalid = False
        except ValueError:
            predicted = []
            invalid = True
        record = {
            "sample_id": str(getattr(state, "sample_id", None) or item["id"]),
            "document_id": source["document_id"],
            "predicted": predicted,
            "target": target_traits,
            "known": list(SCORED_TRAITS),
            "invalid": invalid,
        }
        return Score(
            value=json.dumps(record, ensure_ascii=False),
            explanation=completion,
        )

    return score


def build_arm_task(
    rows: list[dict[str, Any]],
    plan: dict[str, Any],
    fold_number: int,
    arm: str,
) -> tuple[Task, list[str]]:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {tuple(ARMS)}")
    if fold_number < 1 or fold_number > len(plan["folds"]):
        raise ValueError(f"fold_number must be between 1 and {len(plan['folds'])}")
    fold_plan = plan["folds"][fold_number - 1]
    if fold_plan["fold"] != fold_number:
        raise ValueError("Fold plan is not in canonical order")
    include_context, include_demos = ARMS[arm]

    demo_ids = {demo["id"] for demo in fold_plan["demonstrations"]}
    demonstrations = []
    if include_demos:
        by_id = {row["id"]: row for row in rows}
        demonstrations = [
            {
                **demo,
                "text": by_id[demo["id"]]["text"],
            }
            for demo in fold_plan["demonstrations"]
        ]
    validation_ids = set(fold_plan["validation_sample_ids"])
    validation_rows = [
        row for row in rows
        if row["id"] in validation_ids
    ]
    if demo_ids & validation_ids:
        raise ValueError("A validation record was selected as a demonstration")

    samples = []
    for row in validation_rows:
        target_traits = [
            trait for trait in SCORED_TRAITS
            if row["trait_annotations"][trait] == "present"
        ]
        samples.append(Sample(
            input=build_sample_input(row, include_context),
            target=json.dumps(target_traits),
            id=row["id"],
            metadata={
                "item": row,
                "known_traits": list(SCORED_TRAITS),
            },
        ))
    task = Task(
        dataset=MemoryDataset(samples, name=f"conspired-fold-{fold_number}-{arm}"),
        solver=[
            system_message(build_system_prompt(demonstrations)),
            generate(),
        ],
        scorer=trait_prompt_experiment_scorer(),
        name=f"conspired_trait_prompt_{arm}_fold_{fold_number}",
    )
    return task, [sample.id for sample in samples]


def rows_from_eval_log(log: Any) -> list[dict[str, Any]]:
    samples = getattr(log, "samples", None)
    if not samples:
        raise ValueError("Inspect eval log has no sample records")
    rows = []
    for sample in samples:
        scores = getattr(sample, "scores", None) or {}
        score = scores.get("trait_prompt_experiment_scorer")
        if score is None:
            raise ValueError(
                f"Eval sample {getattr(sample, 'id', '<unknown>')!r} "
                "has no trait_prompt_experiment_scorer score"
            )
        try:
            row = json.loads(str(score.value))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid experiment score for sample {getattr(sample, 'id', '<unknown>')!r}"
            ) from exc
        required = {"sample_id", "document_id", "predicted", "target", "known", "invalid"}
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f"Malformed experiment score for sample {getattr(sample, 'id', '<unknown>')!r}")
        if row["sample_id"] != str(getattr(sample, "id", "")):
            raise ValueError(
                f"Experiment score sample ID {row['sample_id']!r} does not match "
                f"Inspect sample {getattr(sample, 'id', '<unknown>')!r}"
            )
        rows.append(row)
    return rows


def _validate_rows(
    rows: list[dict[str, Any]],
    allow_duplicate_sample_ids: bool = False,
) -> None:
    seen_sample_ids: set[str] = set()
    for row in rows:
        if not isinstance(row.get("sample_id"), str) or not row["sample_id"]:
            raise ValueError("Scored row is missing sample_id")
        if row["sample_id"] in seen_sample_ids and not allow_duplicate_sample_ids:
            raise ValueError(f"Duplicate scored sample id: {row['sample_id']}")
        seen_sample_ids.add(row["sample_id"])
        if not isinstance(row.get("document_id"), str) or not row["document_id"]:
            raise ValueError(f"Scored row {row.get('sample_id')!r} is missing document_id")
        if (
            not isinstance(row.get("known"), list)
            or any(
                not isinstance(trait, str) or trait not in SCORED_TRAITS
                for trait in row["known"]
            )
            or set(row["known"]) != set(SCORED_TRAITS)
            or len(row["known"]) != len(SCORED_TRAITS)
        ):
            raise ValueError(f"Scored row {row['sample_id']!r} does not have all six known labels")
        if (
            not isinstance(row.get("target"), list)
            or any(
                not isinstance(trait, str) or trait not in SCORED_TRAITS
                for trait in row["target"]
            )
            or not isinstance(row.get("predicted"), list)
            or any(
                not isinstance(trait, str) or trait not in SCORED_TRAITS
                for trait in row["predicted"]
            )
            or len(row["predicted"]) != len(set(row["predicted"]))
            or len(row["target"]) != len(set(row["target"]))
            or not isinstance(row.get("invalid"), bool)
        ):
            raise ValueError(f"Scored row {row['sample_id']!r} has invalid trait IDs")
        if row.get("invalid") and row["predicted"]:
            raise ValueError(f"Invalid row {row['sample_id']!r} must have no predicted traits")


def metric_summary(
    rows: list[dict[str, Any]],
    allow_duplicate_sample_ids: bool = False,
) -> dict[str, Any]:
    _validate_rows(rows, allow_duplicate_sample_ids=allow_duplicate_sample_ids)
    if not rows:
        raise ValueError("Cannot score an empty row set")
    per_trait = {}
    for trait in SCORED_TRAITS:
        target_positive = sum(trait in row["target"] for row in rows)
        predicted_positive = sum(trait in row["predicted"] for row in rows)
        true_positive = sum(
            trait in row["target"] and trait in row["predicted"] for row in rows
        )
        false_positive = predicted_positive - true_positive
        false_negative = target_positive - true_positive
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / target_positive if target_positive else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if recall is not None and precision + recall
            else 0.0
        )
        per_trait[trait] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "known_count": len(rows),
            "positive_support": target_positive,
            "predicted_positive_count": predicted_positive,
            "false_positive_count": false_positive,
            "false_negative_count": false_negative,
        }

    macro_f1 = sum(values["f1"] for values in per_trait.values()) / len(SCORED_TRAITS)
    total_tp = sum(
        values["positive_support"] - values["false_negative_count"]
        for values in per_trait.values()
    )
    total_predicted = sum(values["predicted_positive_count"] for values in per_trait.values())
    total_target = sum(values["positive_support"] for values in per_trait.values())
    micro_precision = total_tp / total_predicted if total_predicted else 0.0
    micro_recall = total_tp / total_target if total_target else None
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_recall is not None and micro_precision + micro_recall
        else 0.0
    )
    exact_match_count = sum(
        not row["invalid"] and set(row["predicted"]) == set(row["target"])
        for row in rows
    )
    candidate_target = [bool(row["target"]) for row in rows]
    candidate_predicted = [bool(row["predicted"]) for row in rows]
    candidate_tp = sum(target and predicted for target, predicted in zip(candidate_target, candidate_predicted))
    candidate_predicted_count = sum(candidate_predicted)
    candidate_support = sum(candidate_target)
    candidate_precision = candidate_tp / candidate_predicted_count if candidate_predicted_count else 0.0
    candidate_recall = candidate_tp / candidate_support if candidate_support else None
    candidate_f1 = (
        2 * candidate_precision * candidate_recall / (candidate_precision + candidate_recall)
        if candidate_recall is not None and candidate_precision + candidate_recall
        else 0.0
    )
    return {
        "sample_count": len(rows),
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "exact_match": exact_match_count / len(rows),
        "per_trait": per_trait,
        "candidate_detection": {
            "precision": candidate_precision,
            "recall": candidate_recall,
            "f1": candidate_f1,
            "support": candidate_support,
            "predicted_positive_count": candidate_predicted_count,
        },
        "invalid_output_rate": sum(bool(row["invalid"]) for row in rows) / len(rows),
        "predicted_positive_rate_by_trait": {
            trait: per_trait[trait]["predicted_positive_count"] / len(rows)
            for trait in SCORED_TRAITS
        },
        "average_predicted_trait_count": sum(len(row["predicted"]) for row in rows) / len(rows),
        "empty_prediction_rate": sum(not row["predicted"] for row in rows) / len(rows),
        "all_six_prediction_rate": sum(
            set(row["predicted"]) == set(SCORED_TRAITS) for row in rows
        ) / len(rows),
    }


def constant_prediction_rows(
    rows: list[dict[str, Any]], predict_all: bool
) -> list[dict[str, Any]]:
    prediction = list(SCORED_TRAITS) if predict_all else []
    return [
        {
            **row,
            "predicted": prediction,
            "invalid": False,
        }
        for row in rows
    ]


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a quantile without values")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _interval(values: list[float]) -> dict[str, float]:
    return {
        "lower": _quantile(values, 0.025),
        "upper": _quantile(values, 0.975),
    }


def paired_document_bootstrap(
    arm_rows: dict[str, list[dict[str, Any]]],
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = 42,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    if set(arm_rows) != set(ARMS):
        raise ValueError(f"Expected arm rows for exactly {tuple(ARMS)}")
    if any(not rows for rows in arm_rows.values()):
        raise ValueError("Every experiment arm must contain scored rows")
    for rows in arm_rows.values():
        _validate_rows(rows)
    sample_ids = {
        arm: {row["sample_id"] for row in rows}
        for arm, rows in arm_rows.items()
    }
    if len({frozenset(ids) for ids in sample_ids.values()}) != 1:
        raise ValueError("Experiment arms do not contain the same sample IDs")

    rows_by_arm_and_document: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for arm, rows in arm_rows.items():
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(row["document_id"], []).append(row)
        rows_by_arm_and_document[arm] = groups
    reference_rows = {
        row["sample_id"]: (row["document_id"], tuple(sorted(row["target"])))
        for row in arm_rows["snippet_only"]
    }
    for arm, rows in arm_rows.items():
        if {
            row["sample_id"]: (row["document_id"], tuple(sorted(row["target"])))
            for row in rows
        } != reference_rows:
            raise ValueError(f"Arm {arm!r} does not preserve paired targets and documents")
    documents = sorted(next(iter(rows_by_arm_and_document.values())))
    if any(
        set(groups) != set(documents)
        for groups in rows_by_arm_and_document.values()
    ):
        raise ValueError("Experiment arms do not contain the same document groups")

    rng = random.Random(seed)
    arm_values = {arm: [] for arm in ARMS}
    contrasts = {
        "combined_minus_snippet_only": [],
        "combined_minus_all_present": [],
        "context_main_effect": [],
        "demonstrations_main_effect": [],
        "context_demonstrations_interaction": [],
    }
    for _ in range(repetitions):
        sampled_documents = [rng.choice(documents) for _ in documents]
        sampled_rows = {
            arm: [
                row
                for document in sampled_documents
                for row in rows_by_arm_and_document[arm][document]
            ]
            for arm in ARMS
        }
        f1 = {
            arm: metric_summary(rows, allow_duplicate_sample_ids=True)["macro_f1"]
            for arm, rows in sampled_rows.items()
        }
        all_present = metric_summary(
            constant_prediction_rows(sampled_rows["snippet_only"], predict_all=True),
            allow_duplicate_sample_ids=True,
        )["macro_f1"]
        arm_values["snippet_only"].append(f1["snippet_only"])
        arm_values["context_only"].append(f1["context_only"])
        arm_values["demos_only"].append(f1["demos_only"])
        arm_values["context_plus_demos"].append(f1["context_plus_demos"])
        contrasts["combined_minus_snippet_only"].append(
            f1["context_plus_demos"] - f1["snippet_only"]
        )
        contrasts["combined_minus_all_present"].append(
            f1["context_plus_demos"] - all_present
        )
        contrasts["context_main_effect"].append(
            ((f1["context_only"] - f1["snippet_only"])
             + (f1["context_plus_demos"] - f1["demos_only"])) / 2
        )
        contrasts["demonstrations_main_effect"].append(
            ((f1["demos_only"] - f1["snippet_only"])
             + (f1["context_plus_demos"] - f1["context_only"])) / 2
        )
        contrasts["context_demonstrations_interaction"].append(
            f1["context_plus_demos"] - f1["demos_only"]
            - f1["context_only"] + f1["snippet_only"]
        )
    point_f1 = {
        arm: metric_summary(rows)["macro_f1"]
        for arm, rows in arm_rows.items()
    }
    point_all_present = metric_summary(
        constant_prediction_rows(arm_rows["snippet_only"], predict_all=True)
    )["macro_f1"]
    point_contrasts = {
        "combined_minus_snippet_only": (
            point_f1["context_plus_demos"] - point_f1["snippet_only"]
        ),
        "combined_minus_all_present": (
            point_f1["context_plus_demos"] - point_all_present
        ),
        "context_main_effect": (
            (point_f1["context_only"] - point_f1["snippet_only"])
            + (point_f1["context_plus_demos"] - point_f1["demos_only"])
        ) / 2,
        "demonstrations_main_effect": (
            (point_f1["demos_only"] - point_f1["snippet_only"])
            + (point_f1["context_plus_demos"] - point_f1["context_only"])
        ) / 2,
        "context_demonstrations_interaction": (
            point_f1["context_plus_demos"] - point_f1["demos_only"]
            - point_f1["context_only"] + point_f1["snippet_only"]
        ),
    }
    return {
        "arm_macro_f1": {arm: _interval(values) for arm, values in arm_values.items()},
        "contrasts": {
            name: {"estimate": point_contrasts[name], **_interval(values)}
            for name, values in contrasts.items()
        },
        "repetitions": repetitions,
        "seed": seed,
        "resampling_unit": "source.document_id",
        "interval_method": "paired percentile bootstrap with linear interpolation",
    }


def build_experiment_report(
    arm_rows: dict[str, list[dict[str, Any]]],
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = 42,
) -> dict[str, Any]:
    if set(arm_rows) != set(ARMS):
        raise ValueError(f"Expected arm rows for exactly {tuple(ARMS)}")
    for rows in arm_rows.values():
        _validate_rows(rows)
    reference_ids = {row["sample_id"] for row in arm_rows["snippet_only"]}
    for arm, rows in arm_rows.items():
        if {row["sample_id"] for row in rows} != reference_ids:
            raise ValueError(f"Arm {arm!r} does not cover the same validation records")
    summaries = {
        arm: metric_summary(rows)
        for arm, rows in arm_rows.items()
    }
    baselines = {
        "all_six_present": metric_summary(
            constant_prediction_rows(arm_rows["snippet_only"], predict_all=True)
        ),
        "all_six_absent": metric_summary(
            constant_prediction_rows(arm_rows["snippet_only"], predict_all=False)
        ),
    }
    bootstrap = paired_document_bootstrap(arm_rows, repetitions, seed)
    observed_gain = (
        summaries["context_plus_demos"]["macro_f1"]
        - summaries["snippet_only"]["macro_f1"]
    )
    gain_interval = bootstrap["contrasts"]["combined_minus_snippet_only"]
    baseline_interval = bootstrap["contrasts"]["combined_minus_all_present"]
    if gain_interval["upper"] < IMPROVEMENT_THRESHOLD:
        decision = "refuted"
    elif (
        observed_gain >= IMPROVEMENT_THRESHOLD
        and gain_interval["lower"] >= IMPROVEMENT_THRESHOLD
        and baseline_interval["lower"] > 0
    ):
        decision = "supports_claim"
    elif (
        observed_gain >= IMPROVEMENT_THRESHOLD
        and gain_interval["lower"] >= IMPROVEMENT_THRESHOLD
    ):
        decision = "relative_gain_but_classifier_objective_not_met"
    else:
        decision = "inconclusive"
    return {
        "arms": summaries,
        "baselines": baselines,
        "bootstrap": bootstrap,
        "primary_comparison": {
            "combined_arm": "context_plus_demos",
            "reference_arm": "snippet_only",
            "observed_macro_f1_gain": observed_gain,
            "required_gain": IMPROVEMENT_THRESHOLD,
            "paired_95_percent_interval": gain_interval,
            "combined_vs_all_present_95_percent_interval": baseline_interval,
            "decision": decision,
        },
    }
