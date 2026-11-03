"""Tests for CONSPIR trait schema."""
import json
import jsonschema
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent


def test_trait_ids_defined():
    """Test that all 7 CONSPIR traits are defined."""
    trait_ids_file = ROOT / "schemas" / "conspir" / "trait_ids.json"
    assert trait_ids_file.exists()
    
    with open(trait_ids_file) as f:
        data = json.load(f)
    
    assert len(data["traits"]) == 7
    trait_ids = [t["id"] for t in data["traits"]]
    expected = ["contradictory", "overriding_suspicion", "nefarious_intent",
                "something_must_be_wrong", "persecuted_victim", 
                "immune_to_evidence", "reinterpreting_randomness"]
    assert sorted(trait_ids) == sorted(expected)


def test_domain_registry_matches_v2_trait_schema():
    registry = json.loads((ROOT / "schemas" / "conspir" / "domains.json").read_text())
    schema = json.loads((ROOT / "schemas" / "conspir" / "v2_schema.json").read_text())
    expected = {domain["id"] for domain in registry["domains"]} | {registry["unmapped_id"]}
    actual = set(schema["definitions"]["traitRecord"]["properties"]["domain_id"]["enum"])
    assert actual == expected


def test_v2_schema_valid():
    """Test that v2 schema is valid JSON Schema."""
    schema_file = ROOT / "schemas" / "conspir" / "v2_schema.json"
    assert schema_file.exists()
    
    with open(schema_file) as f:
        schema = json.load(f)
    
    # Validate it's a proper schema
    jsonschema.Draft7Validator.check_schema(schema)


def test_trait_record_validation():
    """Test trait record validates against schema."""
    schema_file = ROOT / "schemas" / "conspir" / "v2_schema.json"
    with open(schema_file) as f:
        schema = json.load(f)
    
    trait_record = {
        "schema_version": "2.0",
        "task_type": "trait_classification",
        "id": "trait-test-001",
        "text": "The government is hiding something big.",
        "trait_annotations": {
            "contradictory": "absent",
            "overriding_suspicion": "present",
            "nefarious_intent": "absent",
            "something_must_be_wrong": "present",
            "persecuted_victim": "absent",
            "immune_to_evidence": "absent",
            "reinterpreting_randomness": "absent"
        },
        "source": {"dataset": "test"}
    }
    
    validator = jsonschema.Draft7Validator(schema)
    errors = list(validator.iter_errors(trait_record))
    assert len(errors) == 0, f"Validation errors: {errors}"


def test_unannotated_not_converted_to_absent():
    """Test that unannotated remains unannotated (not converted to absent)."""
    schema_file = ROOT / "schemas" / "conspir" / "v2_schema.json"
    with open(schema_file) as f:
        schema = json.load(f)
    
    trait_record = {
        "schema_version": "2.0",
        "task_type": "trait_classification",
        "id": "trait-test-002",
        "text": "Some text here.",
        "trait_annotations": {
            "contradictory": "absent",
            "overriding_suspicion": "absent",
            "nefarious_intent": "absent",
            "something_must_be_wrong": "unannotated",  # Must remain unannotated
            "persecuted_victim": "absent",
            "immune_to_evidence": "absent",
            "reinterpreting_randomness": "absent"
        },
        "source": {"dataset": "ConspirED"}
    }
    
    validator = jsonschema.Draft7Validator(schema)
    errors = list(validator.iter_errors(trait_record))
    assert len(errors) == 0
    assert trait_record["trait_annotations"]["something_must_be_wrong"] == "unannotated"
