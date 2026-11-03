import pytest

from eval.trait.judge_validation import validate_judgments
from scripts.export_response_judgments import export_rows


def _row(sample_id="cb-1", turn_index=0, **changes):
    row = {
        "sample_id": sample_id,
        "turn_index": turn_index,
        "factual_alignment": "matches",
        "conspiracy_endorsement": "does_not_endorse",
        "abstention": False,
    }
    row.update(changes)
    return row


def test_judge_validation_reports_axis_agreement_and_kappa():
    human = [
        _row(sample_id="cb-1"),
        _row(sample_id="cb-2", factual_alignment="contradicts",
             conspiracy_endorsement="endorses", abstention=True),
    ]
    judge = [
        _row(sample_id="cb-1"),
        _row(sample_id="cb-2", factual_alignment="matches",
             conspiracy_endorsement="endorses", abstention=True),
    ]

    report = validate_judgments(human, judge)
    assert report["compared_units"] == 2
    assert report["axes"]["factual_alignment"]["exact_agreement"] == 0.5
    assert report["axes"]["factual_alignment"]["cohen_kappa"] == 0.0
    assert report["axes"]["abstention"]["exact_agreement"] == 1.0
    assert report["axes"]["abstention"]["cohen_kappa"] == 1.0


def test_judge_validation_reports_human_coverage():
    report = validate_judgments([_row()], [_row(sample_id="cb-2")])
    assert report["compared_units"] == 0
    assert report["human_units_without_valid_judge_output"] == 1
    assert report["valid_judge_units_without_human_reference"] == 1


def test_judge_validation_reports_human_coverage_and_skips_invalid_judge_units():
    human = [_row(), _row(sample_id="cb-2")]
    judge = [_row(), {"sample_id": "cb-2", "turn_index": 0, "invalid": True}]

    report = validate_judgments(human, judge)
    assert report["compared_units"] == 1
    assert report["invalid_judge_units"] == 1
    assert report["human_units_without_valid_judge_output"] == 1


def test_export_response_judgments_keeps_turn_and_epoch_identity():
    import json
    from types import SimpleNamespace

    sample = SimpleNamespace(
        id="cb-1",
        epoch=2,
        scores={
            "response_robustness_scorer": SimpleNamespace(
                value=json.dumps([
                    {"sample_id": "cb-1", "epoch": 2, "turn_index": 0,
                     "factual_alignment": "matches", "conspiracy_endorsement": "endorses",
                     "abstention": False},
                    {"sample_id": "cb-1", "epoch": 2, "turn_index": 1, "invalid": True},
                ])
            )
        },
    )
    log = SimpleNamespace(samples=[sample])
    rows = export_rows(log)
    assert rows[0]["epoch"] == 2
    assert rows[0]["turn_index"] == 0
    assert rows[1]["invalid"] is True


def test_judge_validation_rejects_duplicate_and_invalid_reference_rows():
    with pytest.raises(ValueError, match="Duplicate human judgment"):
        validate_judgments([_row(), _row()], [_row()])
    with pytest.raises(ValueError, match="invalid abstention"):
        validate_judgments([_row(abstention="false")], [_row(abstention="false")])
