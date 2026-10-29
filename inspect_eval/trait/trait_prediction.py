"""Prediction-only CONSPIR run for sources awaiting human trait annotations."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, metric, scorer
from inspect_ai.solver import generate, system_message

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))

from trait.trait_classification import TRAIT_PROMPT, parse_trait_prediction  # noqa: E402


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: expected a JSON object")
                yield row


def _stationhouse_split(group_id: str, seed: int, holdout_fraction: float = 0.30) -> str:
    digest = hashlib.sha256(f"{seed}\0{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "holdout" if value < holdout_fraction else "train"


def load_prediction_items(
    dataset_name: str,
    split: str,
    root: Path = ROOT,
    seed: int = 42,
) -> list[dict]:
    """Load unlabeled text without exposing hidden answers or provenance to the model."""
    root = Path(root)
    if dataset_name == "loco":
        paths = {
            "train": root / "datasets" / "loco" / "imported" / "train_pool.jsonl",
            "holdout": root / "datasets" / "loco" / "imported" / "frozen_holdout.jsonl",
        }
        if split not in paths:
            raise ValueError("LOCO prediction split must be train or holdout")
        records = list(_read_jsonl(paths[split]))
        for record in records:
            if record.get("task_type") != "trait_classification":
                raise ValueError(f"Invalid LOCO trait record {record.get('id')!r}")
        return records
    if dataset_name != "stationhouse":
        raise ValueError("dataset_name must be 'loco' or 'stationhouse'")
    if split not in ("all", "train", "holdout"):
        raise ValueError("Station House prediction split must be all, train, or holdout")

    records = []
    for row in _read_jsonl(root / "questions" / "corpus.jsonl"):
        group_id = str(row.get("claim_id") or row["id"])
        assignment = _stationhouse_split(group_id, seed)
        if split != "all" and assignment != split:
            continue
        records.append({
            "schema_version": "2.0",
            "task_type": "trait_classification",
            "id": f"stationhouse-{row['id']}",
            "text": row["question"],
            "domain_id": "unmapped",
            "trait_annotations": {},
            "source": {
                "dataset": "StationHouse",
                "source_row_id": row["id"],
                "upstream_split": assignment,
            },
            "metadata": {
                "claim_id": row.get("claim_id"),
                "framing": row.get("framing"),
                "legacy_category": row.get("category"),
                "annotation_status": "unannotated",
                "split_seed": seed,
                "split_group_key": "claim_id",
                "split_group": group_id,
            },
        })
    return records


def prediction_dataset(dataset_name: str, split: str, root: Path = ROOT,
                       seed: int = 42) -> MemoryDataset:
    records = load_prediction_items(dataset_name, split, root, seed)
    samples = []
    for record in records:
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Prediction record {record.get('id')!r} has empty text")
        # Deliberately pass no labels, ground-truth answer, source split, or source metadata
        # through the model input; the complete record remains evaluator-side metadata.
        source = record.get("source", {})
        samples.append(Sample(
            input=text,
            target="",
            id=record["id"],
            metadata={
                "item": record,
                "domain_id": record.get("domain_id", "unmapped"),
                "source_dataset": source.get("dataset"),
            },
        ))
    if not samples:
        raise ValueError(f"No records found for dataset={dataset_name!r}, split={split!r}")
    return MemoryDataset(samples, name=f"conspir_trait_predictions_{dataset_name}_{split}")


@metric
def prediction_invalid_output_rate():
    def compute(scores):
        rows = []
        for sample_score in scores:
            try:
                row = json.loads(str(sample_score.score.value))
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                rows.append(row)
        return sum(not row.get("valid", False) for row in rows) / len(rows) if rows else math.nan
    return compute


@scorer(metrics=[prediction_invalid_output_rate()])
def trait_prediction_capture():
    async def score(state, target):
        item = state.metadata["item"]
        source = item.get("source", {})
        sample_id = str(getattr(state, "sample_id", None) or item.get("id", "unknown"))
        row = {
            "sample_id": sample_id,
            "source_dataset": source.get("dataset"),
            "source_row_id": source.get("source_row_id"),
            "document_id": source.get("document_id"),
            "domain_id": item.get("domain_id", "unmapped"),
            "cluster_id": item.get("metadata", {}).get("split_group")
                or source.get("document_id")
                or item.get("metadata", {}).get("claim_id")
                or item.get("id"),
        }
        try:
            row["predicted_traits"] = parse_trait_prediction(state.output.completion)
            row["valid"] = True
            explanation = state.output.completion
        except ValueError as exc:
            row["predicted_traits"] = None
            row["valid"] = False
            row["error"] = str(exc)
            explanation = state.output.completion
        return Score(value=json.dumps(row, ensure_ascii=False), explanation=explanation)
    return score


@task
def conspir_trait_prediction(dataset_name: str = "stationhouse", split: str = "all",
                             seed: int = 42) -> Task:
    """Collect predictions on unannotated LOCO or Station House text.

    This task does not compute trait accuracy. Its output is intended for a
    subsequent blinded human-adjudication workflow.
    """
    return Task(
        dataset=prediction_dataset(dataset_name, split, seed=seed),
        solver=[system_message(TRAIT_PROMPT), generate()],
        scorer=trait_prediction_capture(),
        name=f"conspir_trait_predictions_{dataset_name}_{split}",
    )
