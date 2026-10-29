import json

import jsonschema
import pytest

from scripts.validate_records import LEGACY_SCHEMA, VERSIONED_SCHEMA, schema_for


def test_dispatcher_preserves_legacy_corpus_schema_and_routes_new_records():
    legacy = json.loads((__import__("pathlib").Path(__file__).resolve().parents[2] / "questions" / "corpus.jsonl")
                        .read_text(encoding="utf-8").splitlines()[0])
    assert schema_for(legacy) is LEGACY_SCHEMA

    trait = {
        "schema_version": "2.0", "task_type": "trait_classification", "id": "t1", "text": "text",
        "trait_annotations": {t: "unannotated" for t in (
            "contradictory", "overriding_suspicion", "nefarious_intent", "something_must_be_wrong",
            "persecuted_victim", "immune_to_evidence", "reinterpreting_randomness")},
        "source": {"dataset": "test"},
    }
    assert schema_for(trait) is VERSIONED_SCHEMA
    jsonschema.validate(legacy, LEGACY_SCHEMA)
    jsonschema.validate(trait, VERSIONED_SCHEMA)


def test_response_robustness_record_requires_all_reference_dimensions():
    record = {
        "schema_version": "2.0", "task_type": "response_robustness", "id": "r1",
        "question": "Question?", "ground_truth_answer": "Known answer.", "response": "Model answer.",
        "factual_alignment": "contradicts", "conspiracy_endorsement": "does_not_endorse",
        "abstention": False,
    }
    jsonschema.validate(record, VERSIONED_SCHEMA)
    del record["response"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(record, VERSIONED_SCHEMA)
