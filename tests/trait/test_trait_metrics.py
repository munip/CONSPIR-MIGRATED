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
```
