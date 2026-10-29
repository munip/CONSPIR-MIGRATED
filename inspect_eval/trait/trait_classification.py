"""Trait classification task for CONSPIR."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))

from inspect_ai import Task, task
from inspect_ai.solver import generate, system_message
from inspect_ai.scorer import Score, SampleScore, metric, scorer

from trait.trait_dataset import trait_dataset, TRAIT_IDS


TRAIT_PROMPT = """You are an expert at identifying conspiracy theory traits in text.

Classify the given text for the presence of CONSPIR traits. A text may exhibit zero or more traits.

CONSPIR traits:
1. contradictory - Contradictory claims or internal inconsistencies
2. overriding_suspicion - Overriding suspicion of authority/institutions  
3. nefarious_intent - Attribution of nefarious intent
4. something_must_be_wrong - Something must be wrong/there is a hidden truth
5. persecuted_victim - Self or group portrayed as persecuted victim
6. immune_to_evidence - Claims immune to contrary evidence
7. reinterpreting_randomness - Re-interpreting random events as meaningful

Respond with a JSON array of trait IDs that are present. Only include traits that are clearly exhibited.
Example: ["nefarious_intent", "persecuted_victim"]
If no traits are present, respond with [].
"""


def parse_trait_prediction(completion: str) -> list[str]:
    """Parse a strict JSON array of unique, canonical trait IDs."""
    text = completion.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        closing = text.rfind("```")
        if first_newline < 0 or closing <= first_newline:
            raise ValueError("Unclosed code fence")
        text = text[first_newline + 1:closing].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Output is not valid JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError("Output must be a JSON array")
    if any(not isinstance(trait, str) or trait not in TRAIT_IDS for trait in parsed):
        raise ValueError("Output contains an unknown or non-string trait ID")
    if len(parsed) != len(set(parsed)):
        raise ValueError("Output contains duplicate trait IDs")
    return parsed


def _rows_from_scores(scores: list[SampleScore]):
    for sample_score in scores:
        value = sample_score.score.value
        for epoch_value in (value if isinstance(value, list) else [value]):
            try:
                row = json.loads(str(epoch_value))
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(row, dict) and {"predicted", "target", "known", "invalid"} <= row.keys():
                yield row


def _metric_summary(scores: list[SampleScore]) -> dict:
    rows = list(_rows_from_scores(scores))
    per_trait = {}
    total_tp = total_fp = total_fn = 0
    for trait in TRAIT_IDS:
        tp = fp = fn = known_count = 0
        for row in rows:
            if trait not in row["known"]:
                continue
            known_count += 1
            predicted = trait in row["predicted"]
            target = trait in row["target"]
            tp += predicted and target
            fp += predicted and not target
            fn += not predicted and target
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_trait[trait] = {"precision": precision, "recall": recall, "f1": f1, "known_count": known_count}
        total_tp += tp
        total_fp += fp
        total_fn += fn

    measured = [result for result in per_trait.values() if result["known_count"]]
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if micro_precision + micro_recall else 0.0)
    exact_rows = [row for row in rows if row["known"]]
    exact = sum(
        set(row["predicted"]) & set(row["known"]) == set(row["target"])
        for row in exact_rows
    ) / len(exact_rows) if exact_rows else 0.0
    return {
        "per_trait": per_trait,
        "macro_f1": sum(result["f1"] for result in measured) / len(measured) if measured else 0.0,
        "micro_f1": micro_f1,
        "exact_match": exact,
        "invalid_output_rate": sum(bool(row["invalid"]) for row in rows) / len(rows) if rows else 0.0,
    }


def _trait_metric(name: str, getter):
    def metric_type():
        return lambda scores: getter(_metric_summary(scores))
    metric_type.__name__ = name
    return metric(name=name)(metric_type)()


TRAIT_METRICS = [
    _trait_metric("trait_macro_f1", lambda summary: summary["macro_f1"]),
    _trait_metric("trait_micro_f1", lambda summary: summary["micro_f1"]),
    _trait_metric("trait_exact_match", lambda summary: summary["exact_match"]),
    _trait_metric("trait_invalid_output_rate", lambda summary: summary["invalid_output_rate"]),
]
for _trait_id in TRAIT_IDS:
    for _metric_name in ("precision", "recall", "f1"):
        TRAIT_METRICS.append(_trait_metric(
            f"trait_{_trait_id}_{_metric_name}",
            lambda summary, trait=_trait_id, key=_metric_name: summary["per_trait"][trait][key],
        ))


@scorer(metrics=TRAIT_METRICS)
def trait_scorer():
    async def score(state, target):
        metadata = getattr(state, "metadata", None) or {}
        known_traits = metadata.get("known_traits")
        if known_traits is None:
            annotations = metadata.get("trait_annotations", {})
            known_traits = [
                trait for trait in TRAIT_IDS
                if annotations.get(trait) in ("present", "absent")
            ]
        known_set = set(known_traits)
        if not known_set or not known_set <= set(TRAIT_IDS):
            return Score(
                value=json.dumps({"predicted": [], "target": [], "known": [], "invalid": True}),
                explanation="No valid annotated traits to score",
            )

        target_text = target.text if hasattr(target, "text") else str(target)
        try:
            target_traits = json.loads(target_text)
            if (not isinstance(target_traits, list)
                    or any(trait not in known_set for trait in target_traits)):
                raise ValueError("Invalid trait target for known-label mask")
        except Exception as e:
            record = {"predicted": [], "target": [], "known": sorted(known_set), "invalid": True}
            return Score(value=json.dumps(record), explanation=f"Invalid target: {e}")

        try:
            predicted = parse_trait_prediction(state.output.completion)
            record = {"predicted": predicted, "target": target_traits,
                      "known": sorted(known_set), "invalid": False}
            return Score(value=json.dumps(record), explanation=json.dumps(record))
        except Exception as e:
            record = {"predicted": [], "target": target_traits, "known": sorted(known_set), "invalid": True}
            return Score(value=json.dumps(record), explanation=f"Invalid output: {e}")
    
    return score


@task
def conspir_trait_classification(dataset_name: str = "conspired", split: str = "test") -> Task:
    """CONSPIR trait classification task."""
    return Task(
        dataset=trait_dataset(dataset_name=dataset_name, split=split),
        solver=[system_message(TRAIT_PROMPT), generate()],
        scorer=trait_scorer(),
        name=f"conspir_trait_classification_{dataset_name}_{split}"
    )
