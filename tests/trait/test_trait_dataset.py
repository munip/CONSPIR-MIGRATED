import json
import asyncio
from types import SimpleNamespace

import pytest
from inspect_ai.scorer import Score, SampleScore, Target

from inspect_eval.trait.trait_dataset import trait_to_sample, trait_dataset
from inspect_eval.trait.trait_classification import parse_trait_prediction, _metric_summary, trait_scorer


def test_missing_upstream_trait_is_unannotated_not_absent():
    annotations = {
        "contradictory": "absent",
        "overriding_suspicion": "absent",
        "nefarious_intent": "absent",
        "persecuted_victim": "absent",
        "immune_to_evidence": "absent",
        "reinterpreting_randomness": "absent",
    }
    sample = trait_to_sample({"id": "conspired-1", "text": "sample", "trait_annotations": annotations})

    assert "something_must_be_wrong" not in sample.metadata["known_traits"]
    assert json.loads(sample.target) == []


def test_empty_text_and_invalid_annotation_fail_fast():
    with pytest.raises(ValueError, match="empty text"):
        trait_to_sample({"id": "empty", "text": "  "})
    with pytest.raises(ValueError, match="invalid annotation"):
        trait_to_sample({"id": "bad", "text": "sample", "trait_annotations": {"contradictory": "maybe"}})


def test_trait_dataset_drops_unlabeled_rows_but_keeps_partial_gold(tmp_path):
    conspired = tmp_path / "datasets" / "conspired"
    conspired.mkdir(parents=True)
    (conspired / "normalized.jsonl").write_text(
        json.dumps({"id": "unlabeled", "text": "No gold labels", "trait_annotations": {}}) + "\n"
        + json.dumps({"id": "partial", "text": "Six labels", "trait_annotations": {"nefarious_intent": "present"}}) + "\n",
        encoding="utf-8",
    )

    dataset = trait_dataset(tmp_path)
    assert len(dataset) == 1
    assert dataset[0].id == "partial"


def test_trait_dataset_errors_instead_of_silently_returning_empty(tmp_path):
    with pytest.raises(ValueError, match="No labeled CONSPIR trait rows"):
        trait_dataset(tmp_path)


def test_prediction_parser_rejects_unknown_and_duplicate_ids():
    assert parse_trait_prediction('```json\n["nefarious_intent"]\n```') == ["nefarious_intent"]
    with pytest.raises(ValueError, match="unknown"):
        parse_trait_prediction('["not_a_trait"]')
    with pytest.raises(ValueError, match="duplicate"):
        parse_trait_prediction('["nefarious_intent", "nefarious_intent"]')


def test_inspect_metrics_mask_unknown_traits_and_count_invalid_outputs():
    score = SampleScore(
        sample_id="partial",
        score=Score(value=json.dumps({
            "predicted": ["nefarious_intent", "something_must_be_wrong"],
            "target": ["nefarious_intent"],
            "known": ["nefarious_intent"],
            "invalid": False,
        })),
    )
    invalid = SampleScore(
        sample_id="invalid",
        score=Score(value=json.dumps({"predicted": [], "target": ["nefarious_intent"], "known": ["nefarious_intent"], "invalid": True})),
    )

    summary = _metric_summary([score, invalid])
    assert summary["per_trait"]["nefarious_intent"]["f1"] == pytest.approx(2 / 3)
    assert summary["exact_match"] == 0.5
    assert summary["invalid_output_rate"] == 0.5


def test_inspect_scorer_ignores_unannotated_predictions_and_flags_unknown_ids():
    score_fn = trait_scorer()
    state = SimpleNamespace(
        metadata={"known_traits": ["nefarious_intent"]},
        output=SimpleNamespace(completion='["something_must_be_wrong"]'),
    )
    score = asyncio.run(score_fn(state, Target("[]")))
    record = json.loads(score.value)
    assert record["invalid"] is False
    assert record["known"] == ["nefarious_intent"]

    state.output.completion = '["not_a_trait"]'
    invalid_score = asyncio.run(score_fn(state, Target("[]")))
    assert json.loads(invalid_score.value)["invalid"] is True
