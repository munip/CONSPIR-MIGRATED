import json

import pytest

from inspect_eval.trait.trait_prediction import (
    _stationhouse_split,
    load_prediction_items,
    prediction_dataset,
)


def test_stationhouse_prediction_loader_splits_by_claim_and_uses_question_only(tmp_path):
    questions = tmp_path / "questions"
    questions.mkdir()
    rows = [
        {"id": "cb-1", "claim_id": "same-claim", "question": "Question one?",
         "ground_truth_answer": "Hidden answer one.", "category": "space"},
        {"id": "cb-2", "claim_id": "same-claim", "question": "Question two?",
         "ground_truth_answer": "Hidden answer two.", "category": "space"},
        {"id": "cb-3", "question": "Question three?", "ground_truth_answer": "Hidden answer three."},
    ]
    (questions / "corpus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    assignments = {
        row["id"]: _stationhouse_split(str(row.get("claim_id") or row["id"]), seed=4)
        for row in rows
    }
    holdout = load_prediction_items("stationhouse", "holdout", tmp_path, seed=4)
    selected = {row["source"]["source_row_id"] for row in holdout}
    assert selected == {key for key, value in assignments.items() if value == "holdout"}
    assert ("cb-1" in selected) == ("cb-2" in selected)
    dataset = prediction_dataset("stationhouse", "all", tmp_path, seed=4)
    assert all("Hidden answer" not in sample.input for sample in dataset)
    assert all(sample.metadata["item"]["trait_annotations"] == {} for sample in dataset)


def test_loco_prediction_loader_uses_unannotated_imported_records(tmp_path):
    imported = tmp_path / "datasets" / "loco" / "imported"
    imported.mkdir(parents=True)
    record = {
        "id": "loco-1", "task_type": "trait_classification", "text": "Document text.",
        "domain_id": "unmapped", "trait_annotations": {"nefarious_intent": "unannotated"},
        "source": {"dataset": "LOCO", "document_id": "doc-1"},
    }
    (imported / "train_pool.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (imported / "frozen_holdout.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    rows = load_prediction_items("loco", "holdout", tmp_path)
    assert rows[0]["id"] == "loco-1"
    assert rows[0]["trait_annotations"]["nefarious_intent"] == "unannotated"
    sample = prediction_dataset("loco", "holdout", tmp_path)[0]
    assert sample.input == "Document text."
    assert sample.target == ""


def test_stationhouse_prediction_loader_rejects_bad_split(tmp_path):
    (tmp_path / "questions").mkdir()
    (tmp_path / "questions" / "corpus.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="split must be all"):
        load_prediction_items("stationhouse", "test", tmp_path)
