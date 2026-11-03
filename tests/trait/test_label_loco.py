import json
from pathlib import Path

import pytest

from scripts import label_loco
from scripts.evaluate_trait_labeler import metrics


def _record(dataset, split, annotations=None, text="sample text"):
    annotations = annotations or {trait: "unannotated" for trait in label_loco.TRAITS}
    annotations.setdefault(label_loco.UNLABELED, "unannotated")
    return {
        "schema_version": "2.0",
        "task_type": "trait_classification",
        "id": f"{dataset}-{split}",
        "text": text,
        "trait_annotations": annotations,
        "present_traits": [],
        "known_traits": [],
        "source": {"dataset": dataset, "upstream_split": split},
        "metadata": {"label_status": "unannotated"},
    }


def _write_jsonl(path: Path, row):
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_parse_prediction_accepts_only_unique_known_traits():
    assert label_loco.parse_prediction('["nefarious_intent"]') == ["nefarious_intent"]
    with pytest.raises(ValueError):
        label_loco.parse_prediction('["made_up_trait"]')
    with pytest.raises(ValueError):
        label_loco.parse_prediction('["nefarious_intent", "nefarious_intent"]')


def test_evaluation_metrics_are_multi_label_and_include_exact_match():
    result = metrics(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
        [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
    )
    assert result["exact_match"] == 0.5
    assert result["micro_f1"] == pytest.approx(2 / 3)


def test_word_windows_overlap_without_empty_trailing_window():
    words = " ".join(f"w{i}" for i in range(300))
    chunks = label_loco.word_windows(words, size=100, overlap=20)
    assert len(chunks) == 4
    assert len(chunks[0].split()) == 100
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_loco_labeler_refuses_holdout(tmp_path, monkeypatch):
    train = tmp_path / "conspired-train.jsonl"
    loco = tmp_path / "loco-holdout.jsonl"
    output = tmp_path / "out.jsonl"
    _write_jsonl(train, _record("ConspirED", "train", {
        **{trait: "absent" for trait in label_loco.TRAITS},
        label_loco.UNLABELED: "unannotated",
    }))
    _write_jsonl(loco, _record("LOCO", "holdout"))
    monkeypatch.setattr(label_loco, "train_supervised", lambda rows: object())
    with pytest.raises(ValueError, match="refusing to label rows outside"):
        label_loco.label_loco("supervised", train, loco, output)


def test_candidate_labels_are_marked_as_pseudo_labels(tmp_path, monkeypatch):
    train = tmp_path / "conspired-train.jsonl"
    loco = tmp_path / "loco-train.jsonl"
    output = tmp_path / "candidates.jsonl"
    train_row = _record("ConspirED", "train", {
        **{trait: "absent" for trait in label_loco.TRAITS},
        label_loco.UNLABELED: "unannotated",
    })
    _write_jsonl(train, train_row)
    _write_jsonl(loco, _record("LOCO", "train", text="Long article text."))

    # Return one score column per trait, matching sklearn's multilabel output.
    class FakeModel:
        def predict_proba(self, chunks):
            import numpy as np
            return np.array([[0.9] + [0.1] * (len(label_loco.TRAITS) - 1) for _ in chunks])

    monkeypatch.setattr(label_loco, "train_supervised", lambda rows: FakeModel())
    summary = label_loco.label_loco("supervised", train, loco, output)
    result = next(label_loco.read_jsonl(output))
    assert summary["input_records"] == 1
    assert result["trait_annotations"][label_loco.TRAITS[0]] == "present"
    assert result["trait_annotations"][label_loco.UNLABELED] == "unannotated"
    assert result["metadata"]["label_status"] == "pseudo_labeled"
    assert result["metadata"]["candidate_annotation"]["gold_label"] is False


def test_invalid_qwen_output_remains_unannotated(tmp_path, monkeypatch):
    train = tmp_path / "conspired-train.jsonl"
    loco = tmp_path / "loco-train.jsonl"
    output = tmp_path / "candidates.jsonl"
    _write_jsonl(train, _record("ConspirED", "train", {
        **{trait: "absent" for trait in label_loco.TRAITS},
        label_loco.UNLABELED: "unannotated",
    }))
    _write_jsonl(loco, _record("LOCO", "train"))
    monkeypatch.setattr(label_loco, "qwen_predictor", lambda *args: lambda text: (_ for _ in ()).throw(ValueError("bad JSON")))
    label_loco.label_loco("qwen", train, loco, output, model_id="local-model")
    result = next(label_loco.read_jsonl(output))
    assert all(result["trait_annotations"][trait] == "unannotated" for trait in label_loco.TRAITS)
    assert result["known_traits"] == []
    assert result["metadata"]["label_status"] == "invalid"
