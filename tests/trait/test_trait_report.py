import json
from types import SimpleNamespace

import pytest
from inspect_ai.scorer import Score

from inspect_eval.trait.trait_report import report_from_log


def test_trait_report_includes_grouped_metrics_and_serializable_unsupported_values():
    rows = [
        {
            "predicted": ["nefarious_intent"],
            "target": ["nefarious_intent"],
            "known": ["nefarious_intent"],
            "invalid": False,
            "domain_id": "health_medicine",
            "source_dataset": "StationHouse",
            "cluster_id": "claim-1",
        },
    ]
    log = SimpleNamespace(samples=[
        SimpleNamespace(
            scores={"trait_scorer": Score(value=json.dumps(rows[0]))},
        ),
    ])

    report = report_from_log(log)
    assert report["overall"]["invalid_output_rate"] == 0.0
    assert report["by_domain"]["health_medicine"]["per_trait"]["nefarious_intent"]["f1"] == 1.0
    assert report["by_source_reference_support"]["StationHouse"]["nefarious_intent"]["positive_support"] == 1
    assert report["overall"]["per_trait"]["something_must_be_wrong"]["f1"] is None


def test_trait_report_requires_scored_samples():
    with pytest.raises(ValueError, match="contains no 'trait_scorer' scores"):
        report_from_log(SimpleNamespace(samples=[]))
