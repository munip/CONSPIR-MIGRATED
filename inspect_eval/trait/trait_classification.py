"""Trait classification task for CONSPIR."""
from __future__ import annotations

import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
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


def _trait_counts(rows: list[dict]) -> dict:
    per_trait = {}
    total_tp = total_fp = total_fn = 0
    for trait in TRAIT_IDS:
        tp = fp = fn = known_count = positive_support = 0
        for row in rows:
            if trait not in row["known"]:
                continue
            known_count += 1
            predicted = trait in row["predicted"]
            target = trait in row["target"]
            positive_support += target
            tp += predicted and target
            fp += predicted and not target
            fn += not predicted and target
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else math.nan
        if not positive_support:
            f1 = math.nan
        else:
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_trait[trait] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "known_count": known_count,
            "positive_support": positive_support,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
    measured = [result for result in per_trait.values() if result["positive_support"]]
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    if not total_tp + total_fn:
        micro_f1 = math.nan
    else:
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall else 0.0
        )
    exact_rows = [row for row in rows if set(row["known"]) == set(TRAIT_IDS)]
    exact = sum(
        set(row["predicted"]) & set(row["known"]) == set(row["target"])
        for row in exact_rows
    ) / len(exact_rows) if exact_rows else math.nan
    return {
        "per_trait": per_trait,
        "macro_f1": sum(result["f1"] for result in measured) / len(measured) if measured else math.nan,
        "micro_f1": micro_f1,
        "exact_match": exact,
    }


def _reference_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    return {
        trait: {
            "known_count": sum(trait in row["known"] for row in rows),
            "positive_support": sum(
                trait in row["known"] and trait in row["target"] for row in rows
            ),
        }
        for trait in TRAIT_IDS
    }


def _cluster_bootstrap_f1(rows: list[dict], trait: str, repetitions: int = 500) -> tuple[float, float]:
    clusters: dict[str, list[dict]] = {}
    for row in rows:
        if trait in row["known"]:
            clusters.setdefault(str(row.get("cluster_id") or row.get("sample_id") or len(clusters)), []).append(row)
    if not clusters:
        return math.nan, math.nan
    cluster_rows = list(clusters.values())
    positive_clusters = sum(
        any(trait in row["target"] for row in cluster)
        for cluster in cluster_rows
    )
    if positive_clusters < 5:
        return math.nan, math.nan
    rng = random.Random(f"conspir-trait-ci:{trait}")
    estimates = []
    for _ in range(repetitions):
        sample = [row for _ in cluster_rows for row in rng.choice(cluster_rows)]
        estimate = _trait_counts(sample)["per_trait"][trait]["f1"]
        if math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return math.nan, math.nan
    estimates.sort()
    return estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]


def _group_summaries(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            groups.setdefault(str(value), []).append(row)
    return {group: _trait_counts(group_rows) for group, group_rows in groups.items()}


def _group_reference_counts(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            groups.setdefault(str(value), []).append(row)
    return {group: _reference_counts(group_rows) for group, group_rows in groups.items()}


@lru_cache(maxsize=8)
def _cached_metric_summary(values: tuple[str, ...]) -> dict:
    rows = [json.loads(value) for value in values]
    valid_rows = [row for row in rows if not row.get("invalid")]
    summary = _trait_counts(valid_rows)
    reference_counts = _reference_counts(rows)
    for trait in TRAIT_IDS:
        summary["per_trait"][trait]["reference_known_count"] = reference_counts[trait]["known_count"]
        summary["per_trait"][trait]["reference_positive_support"] = reference_counts[trait]["positive_support"]
        summary["per_trait"][trait]["f1_ci_low"], summary["per_trait"][trait]["f1_ci_high"] = (
            _cluster_bootstrap_f1(valid_rows, trait)
        )
    dominant = [
        row for row in valid_rows
        if row.get("dominant_trait") in row["known"]
    ]
    summary["dominant_trait_hit_rate"] = (
        sum(row["dominant_trait"] in row["predicted"] for row in dominant) / len(dominant)
        if dominant else math.nan
    )
    summary["dominant_trait_support"] = len(dominant)
    summary["dominant_trait_reference_support"] = sum(
        row.get("dominant_trait") in row["known"] for row in rows
    )
    summary["by_domain"] = _group_summaries(valid_rows, "domain_id")
    summary["by_source"] = _group_summaries(valid_rows, "source_dataset")
    summary["by_domain_reference"] = _group_reference_counts(rows, "domain_id")
    summary["by_source_reference"] = _group_reference_counts(rows, "source_dataset")
    summary["invalid_output_rate"] = (
        sum(bool(row.get("invalid")) for row in rows) / len(rows) if rows else math.nan
    )
    summary["valid_output_count"] = len(valid_rows)
    summary["total_output_count"] = len(rows)
    return summary


def _metric_summary(scores: list[SampleScore]) -> dict:
    values = tuple(
        json.dumps(row, sort_keys=True, ensure_ascii=False)
        for row in _rows_from_scores(scores)
    )
    return _cached_metric_summary(values)


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
    _trait_metric("trait_valid_output_count", lambda summary: summary["valid_output_count"]),
    _trait_metric("trait_total_output_count", lambda summary: summary["total_output_count"]),
    _trait_metric("trait_dominant_hit_rate", lambda summary: summary["dominant_trait_hit_rate"]),
    _trait_metric("trait_dominant_support", lambda summary: summary["dominant_trait_support"]),
    _trait_metric("trait_dominant_reference_support", lambda summary: summary["dominant_trait_reference_support"]),
]
for _trait_id in TRAIT_IDS:
    for _metric_name in (
        "precision", "recall", "f1", "known_count", "positive_support",
        "f1_ci_low", "f1_ci_high",
    ):
        TRAIT_METRICS.append(_trait_metric(
            f"trait_{_trait_id}_{_metric_name}",
            lambda summary, trait=_trait_id, key=_metric_name: summary["per_trait"][trait][key],
        ))
    for _metric_name in ("reference_known_count", "reference_positive_support"):
        TRAIT_METRICS.append(_trait_metric(
            f"trait_{_trait_id}_{_metric_name}",
            lambda summary, trait=_trait_id, key=_metric_name: summary["per_trait"][trait][key],
        ))

@scorer(metrics=TRAIT_METRICS)
def trait_scorer():
    async def score(state, target):
        metadata = getattr(state, "metadata", None) or {}
        item = metadata.get("item", {})
        source = item.get("source", {})
        source_dataset = source.get("dataset") if isinstance(source, dict) else None
        cluster_id = (
            source.get("document_id") if isinstance(source, dict) else None
        ) or item.get("claim_id") or item.get("metadata", {}).get("claim_id") or \
            item.get("metadata", {}).get("split_group") or item.get("id") or metadata.get("sample_id")
        record_metadata = {
            "domain_id": item.get("domain_id", "unmapped"),
            "source_dataset": source_dataset,
            "cluster_id": cluster_id,
            "sample_id": metadata.get("sample_id"),
            "dominant_trait": item.get("dominant_trait"),
        }
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
                value=json.dumps({
                    "predicted": [], "target": [], "known": [], "invalid": True,
                    **record_metadata,
                }),
                explanation="No valid annotated traits to score",
            )

        target_text = target.text if hasattr(target, "text") else str(target)
        try:
            target_traits = json.loads(target_text)
            if (not isinstance(target_traits, list)
                    or any(not isinstance(trait, str) or trait not in known_set
                           for trait in target_traits)):
                raise ValueError("Invalid trait target for known-label mask")
        except ValueError as e:
            record = {
                "predicted": [], "target": [], "known": sorted(known_set), "invalid": True,
                **record_metadata,
            }
            return Score(value=json.dumps(record), explanation=f"Invalid target: {e}")

        try:
            predicted = parse_trait_prediction(state.output.completion)
            record = {"predicted": predicted, "target": target_traits,
                      "known": sorted(known_set), "invalid": False, **record_metadata}
            return Score(value=json.dumps(record), explanation=json.dumps(record))
        except ValueError as e:
            record = {
                "predicted": [], "target": target_traits, "known": sorted(known_set),
                "invalid": True, **record_metadata,
            }
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
