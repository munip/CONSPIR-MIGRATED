import json
from decimal import Decimal

import jsonschema
import pytest

from scripts.import_loco import import_loco
from scripts.validate_records import VERSIONED_SCHEMA


def _write_loco(path, rows):
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_loco_import_preserves_provenance_and_freezes_topic_groups(tmp_path):
    source = tmp_path / "LOCO.json"
    rows = [
        {"doc_id": "a1", "txt": "text a1", "topic_k100": 1, "subcorpus": "conspiracy", "URL": "https://a"},
        {"doc_id": "a2", "txt": "text a2", "topic_k100": 1, "subcorpus": "mainstream", "URL": "https://b"},
        {"doc_id": "b1", "txt": "text b1", "topic_k100": 2, "subcorpus": "mainstream"},
        {"doc_id": "b2", "txt": "text b2", "topic_k100": 2, "subcorpus": "conspiracy"},
        {"doc_id": "c1", "txt": "text c1", "topic_k100": 3, "subcorpus": "mainstream"},
        {"doc_id": "d1", "txt": "text d1", "topic_k100": 4, "subcorpus": "conspiracy"},
    ]
    _write_loco(source, rows)
    output = tmp_path / "out"

    manifest = import_loco(source, output, seed=9, holdout_fraction=0.33, source_revision="fixture-sha")
    train = [json.loads(line) for line in (output / "train_pool.jsonl").read_text(encoding="utf-8").splitlines()]
    holdout = [json.loads(line) for line in (output / "frozen_holdout.jsonl").read_text(encoding="utf-8").splitlines()]

    assert train and holdout
    train_groups = {row["metadata"]["split_group"] for row in train}
    holdout_groups = {row["metadata"]["split_group"] for row in holdout}
    assert train_groups.isdisjoint(holdout_groups)
    assert manifest["documents"] == len(rows)
    assert all(row["source"]["dataset"] == "LOCO" for row in train + holdout)
    assert all(row["trait_annotations"]["something_must_be_wrong"] == "unannotated" for row in train + holdout)
    assert {row["metadata"]["subcorpus"] for row in train + holdout} == {"conspiracy", "mainstream"}
    for row in train + holdout:
        jsonschema.validate(row, VERSIONED_SCHEMA)


def test_loco_import_fails_on_duplicate_documents_or_missing_topic_groups(tmp_path):
    source = tmp_path / "LOCO.json"
    row = {"doc_id": "same", "txt": "text", "topic_k100": 1}
    _write_loco(source, [row, row])
    with pytest.raises(ValueError, match="Duplicate LOCO doc_id"):
        import_loco(source, tmp_path / "duplicates")

    _write_loco(source, [{"doc_id": "x", "txt": "text"}])
    with pytest.raises(ValueError, match="missing topic_k100"):
        import_loco(source, tmp_path / "missing-topic")


def test_loco_import_serializes_decimal_metadata_from_streaming_parser(tmp_path):
    source = tmp_path / "LOCO.json"
    rows = [
        {"doc_id": "a", "txt": "text a", "topic_k100": 1, "cosine_similarity": 0.14},
        {"doc_id": "b", "txt": "text b", "topic_k100": 2, "cosine_similarity": 0.24},
    ]
    _write_loco(source, rows)
    output = tmp_path / "decimal-out"

    import_loco(source, output)
    records = [
        json.loads(line)
        for name in ("train_pool.jsonl", "frozen_holdout.jsonl")
        for line in (output / name).read_text(encoding="utf-8").splitlines()
    ]
    assert {row["metadata"]["source_metadata"]["cosine_similarity"] for row in records} == {0.14, 0.24}


def test_loco_import_joins_topic_and_website_sidecars_without_creating_trait_labels(tmp_path):
    source = tmp_path / "LOCO.json"
    rows = [
        {"doc_id": "a", "txt": "text a", "topic_k100": "k100_1", "website": "example.org"},
        {"doc_id": "b", "txt": "text b", "topic_k100": "k100_2", "website": "other.org"},
    ]
    _write_loco(source, rows)
    topics = tmp_path / "topic_description.json"
    topics.write_text(json.dumps([{"Topic_ID": "k100_1", "Top_15_words": "science health"}]), encoding="utf-8")
    websites = tmp_path / "website_metadata.json"
    websites.write_text(json.dumps([{"website": "example.org", "MBFC_factual_reporting": "high"}]), encoding="utf-8")
    output = tmp_path / "joined"

    manifest = import_loco(
        source, output, topic_description_path=topics, website_metadata_path=websites
    )
    records = [
        json.loads(line)
        for name in ("train_pool.jsonl", "frozen_holdout.jsonl")
        for line in (output / name).read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["documents"] == 2
    assert manifest["documents_with_topic_description"] == 1
    assert manifest["documents_with_website_metadata"] == 1
    assert all(all(value == "unannotated" for value in row["trait_annotations"].values()) for row in records)
    joined = next(row for row in records if row["source"]["document_id"] == "a")
    assert joined["metadata"]["topic_description"]["Top_15_words"] == "science health"
    assert joined["metadata"]["website_metadata"]["MBFC_factual_reporting"] == "high"
    for row in records:
        jsonschema.validate(row, VERSIONED_SCHEMA)
