import json

import jsonschema
import pytest

openpyxl = pytest.importorskip("openpyxl")

from scripts.import_conspired import import_conspired
from scripts.validate_records import VERSIONED_SCHEMA


HEADERS = [
    "doc_id", "id", "snippet", "context500", "context1000", "labels",
    "consolidated_trait", "dominant_consol_trait", "single_dominant_one_hot_dm",
    "Justification", "Confidence", "ConspiracyTheoryorMainstream", "begin", "end",
]


def _write_xlsx(path, row):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(HEADERS)
    sheet.append([row.get(header) for header in HEADERS])
    workbook.save(path)


def test_conspired_import_preserves_splits_maps_six_labels_and_keeps_missing_trait_unknown(tmp_path):
    train = tmp_path / "train.xlsx"
    test = tmp_path / "test.xlsx"
    _write_xlsx(train, {
        "doc_id": "doc-train", "id": "a-1", "snippet": "A labeled snippet.",
        "context500": "Short context.", "context1000": "Long context.",
        "labels": "[0, 1, 1, 0, 0, 0]",
        "consolidated_trait": "['Overriding suspicion', 'Nefarious intent']",
        "dominant_consol_trait": "['Nefarious intent']",
        "single_dominant_one_hot_dm": "[0, 0, 1, 0, 0, 0]",
        "Justification": "The author attributes hidden motives.", "Confidence": "[3]",
        "ConspiracyTheoryorMainstream": "[None]", "begin": 10, "end": 30,
    })
    _write_xlsx(test, {
        "doc_id": "doc-test", "id": "a-1", "snippet": "Held-out snippet.",
        "labels": "[0, 0, 1, 0, 0, 0]", "consolidated_trait": "['Nefarious intent']",
        "dominant_consol_trait": "['Overriding suspicion']",
        "single_dominant_one_hot_dm": "[0, 0, 1, 0, 0, 0]",
    })
    output = tmp_path / "out"

    manifest = import_conspired(train, test, output, source_revision="fixture-sha", context_column="context1000")
    train_record = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    test_record = json.loads((output / "test.jsonl").read_text(encoding="utf-8"))

    assert train_record["source"]["upstream_split"] == "train"
    assert test_record["source"]["upstream_split"] == "test"
    assert train_record["context"] == "Long context."
    assert train_record["trait_annotations"]["overriding_suspicion"] == "present"
    assert train_record["trait_annotations"]["contradictory"] == "absent"
    assert train_record["trait_annotations"]["something_must_be_wrong"] == "unannotated"
    assert train_record["dominant_trait"] == "nefarious_intent"
    assert train_record["metadata"]["conspiracy_theory_or_mainstream_raw"] == "[None]"
    assert test_record["metadata"]["dominant_label_conflict"] is True
    assert train_record["id"] != test_record["id"]
    assert manifest["train_test_document_overlap"] == 0
    assert manifest["dominant_consol_trait_conflicts_with_one_hot"]["test"] == 1
    for record in (train_record, test_record):
        jsonschema.validate(record, VERSIONED_SCHEMA)


def test_conspired_import_rejects_misaligned_trait_vector_and_document_leakage(tmp_path):
    train = tmp_path / "train.xlsx"
    test = tmp_path / "test.xlsx"
    row = {
        "doc_id": "same-doc", "id": "a-1", "snippet": "A labeled snippet.",
        "labels": "[1, 0, 0, 0, 0, 0]", "consolidated_trait": "['Nefarious intent']",
        "single_dominant_one_hot_dm": "[1, 0, 0, 0, 0, 0]",
    }
    _write_xlsx(train, row)
    _write_xlsx(test, {**row, "snippet": "Same source document, test row."})
    with pytest.raises(ValueError, match="labels disagree"):
        import_conspired(train, test, tmp_path / "mismatch")

    row["consolidated_trait"] = "['Contradictory']"
    _write_xlsx(train, row)
    _write_xlsx(test, {**row, "id": "b-1"})
    with pytest.raises(ValueError, match="share 1 document IDs"):
        import_conspired(train, test, tmp_path / "leakage")
