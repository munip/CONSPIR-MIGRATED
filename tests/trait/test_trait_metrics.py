"""Tests for trait metrics."""
from eval.trait.trait_metrics import compute_trait_metrics, TRAIT_IDS


def test_compute_trait_metrics():
    """Test trait metrics computation with known-label masking."""
    predictions = {
        "item1": ["nefarious_intent", "overriding_suspicion"],
        "item2": ["immune_to_evidence"]
    }
    references = {
        "item1": {
            "contradictory": "absent",
            "overriding_suspicion": "present",
            "nefarious_intent": "present",
            "something_must_be_wrong": "unannotated",  # Should be masked
            "persecuted_victim": "absent",
            "immune_to_evidence": "absent",
            "reinterpreting_randomness": "absent"
        },
        "item2": {
            "contradictory": "absent",
            "overriding_suspicion": "absent",
            "nefarious_intent": "absent",
            "something_must_be_wrong": "absent",
            "persecuted_victim": "absent",
            "immune_to_evidence": "present",
            "reinterpreting_randomness": "absent"
        }
    }
    
    results = compute_trait_metrics(predictions, references)
    assert "per_trait" in results
    assert "macro" in results
    assert "micro" in results
    # something_must_be_wrong should not penalize for being unannotated
    assert results["per_trait"]["nefarious_intent"]["f1"] == 1.0
    assert results["exact_match"] == 1.0
    assert results["invalid_output_rate"] == 0.0


def test_exact_match_ignores_predictions_for_unannotated_labels():
    prediction = {"item": ["nefarious_intent", "something_must_be_wrong"]}
    reference = {"item": {"nefarious_intent": "present", "something_must_be_wrong": "unannotated"}}

    results = compute_trait_metrics(prediction, reference)
    assert results["exact_match"] == 1.0


def test_invalid_output_rate_uses_explicit_run_counts():
    results = compute_trait_metrics({}, {}, invalid_outputs=2, total_outputs=5)
    assert results["invalid_output_rate"] == 0.4
