import json
from decimal import Decimal

import jsonschema
import pytest

from scripts.import_loco import import_loco
from scripts.propose_loco_domains import propose_domains
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


def test_loco_domain_proposals_require_review_before_assigning_domain(tmp_path):
    source = tmp_path / "LOCO.json"
    _write_loco(source, [
        {"doc_id": "a", "txt": "health text", "topic_k100": "k100_1"},
        {"doc_id": "b", "txt": "unmatched text", "topic_k100": "k100_2"},
    ])
    topics = tmp_path / "topic_description.json"
    topics.write_text(json.dumps([
        {"Topic_ID": "k100_1", "Top_15_words": "health medicine vaccine"},
        {"Topic_ID": "k100_2", "Top_15_words": "unrelated tokens"},
        {"Topic_ID": "k200_1", "Top_15_words": "must not enter k100 mapping"},
    ]), encoding="utf-8")
    mapping_path = tmp_path / "domain_mapping.json"
    summary = propose_domains(topics, mapping_path)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert summary["topics"] == 2
    assert mapping["topics"][0]["suggested_domain_id"] == "health_medicine"
    assert mapping["topics"][0]["status"] == "pending_review"
    assert mapping["topics"][0]["domain_id"] is None
    assert mapping["topics"][1]["suggested_domain_id"] is None

    output = tmp_path / "out"
    import_loco(
        source, output, topic_description_path=topics,
        domain_mapping_path=mapping_path,
    )
    records = [
        json.loads(line)
        for name in ("train_pool.jsonl", "frozen_holdout.jsonl")
        for line in (output / name).read_text(encoding="utf-8").splitlines()
    ]
    mapped_suggestion = next(row for row in records if row["source"]["document_id"] == "a")
    assert mapped_suggestion["domain_id"] == "unmapped"
    assert mapped_suggestion["metadata"]["suggested_domain_id"] == "health_medicine"
    assert mapped_suggestion["metadata"]["domain_mapping_status"] == "pending_review"

    mapping["topics"][0]["domain_id"] = "health_medicine"
    mapping["topics"][0]["status"] = "approved"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    approved_output = tmp_path / "approved"
    manifest = import_loco(
        source, approved_output, topic_description_path=topics,
        domain_mapping_path=mapping_path,
    )
    approved_records = [
        json.loads(line)
        for name in ("train_pool.jsonl", "frozen_holdout.jsonl")
        for line in (approved_output / name).read_text(encoding="utf-8").splitlines()
    ]
    approved = next(row for row in approved_records if row["source"]["document_id"] == "a")
    assert approved["domain_id"] == "health_medicine"
    assert manifest["documents_with_approved_domain_mapping"] == 1

    topics.write_text(json.dumps([
        {"Topic_ID": "k100_1", "Top_15_words": "changed source"},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="different topic descriptions"):
        import_loco(
            source, tmp_path / "stale-source", topic_description_path=topics,
            domain_mapping_path=mapping_path,
        )


def test_loco_import_rejects_duplicate_sidecar_and_unregistered_domain(tmp_path):
    source = tmp_path / "LOCO.json"
    _write_loco(source, [
        {"doc_id": "a", "txt": "text", "topic_k100": "k100_1"},
        {"doc_id": "b", "txt": "text", "topic_k100": "k100_2"},
    ])
    sidecar = tmp_path / "topics.json"
    sidecar.write_text(json.dumps([
        {"Topic_ID": "k100_1"}, {"Topic_ID": "k100_1"},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate Topic_ID"):
        import_loco(source, tmp_path / "duplicate-sidecar", topic_description_path=sidecar)

    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"topics": [
        {"topic_id": "k100_1", "status": "approved", "domain_id": "made_up"},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid domain_id"):
        import_loco(source, tmp_path / "bad-domain", domain_mapping_path=mapping)
