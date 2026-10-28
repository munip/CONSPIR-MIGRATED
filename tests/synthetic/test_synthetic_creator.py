"""Tests for synthetic data creator."""
import json
import tempfile
from pathlib import Path
from pipeline.synthetic.conspir_synthetic_creator import ConspirSyntheticCreator, TRAIT_IDS


def test_create_candidate():
    """Test creating a synthetic candidate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        creator = ConspirSyntheticCreator(Path(tmpdir))
        candidate = creator.create_candidate(
            text="They are secretly controlling everything from behind the scenes.",
            requested_traits=["nefarious_intent", "overriding_suspicion"],
            domain_id="politics",
            model="test-model"
        )
        
        assert candidate["schema_version"] == "2.0"
        assert candidate["task_type"] == "trait_classification"
        assert candidate["domain_id"] == "politics"
        assert candidate["requested_traits"] == ["nefarious_intent", "overriding_suspicion"]
        assert candidate["review_state"] == "pending"
        assert all(candidate["trait_annotations"][t] == "unannotated" for t in TRAIT_IDS)


def test_review_process():
    """Test blind review process."""
    with tempfile.TemporaryDirectory() as tmpdir:
        creator = ConspirSyntheticCreator(Path(tmpdir))
        candidate = creator.create_candidate(
            text="Test text",
            requested_traits=["persecuted_victim"],
            domain_id="health"
        )
        
        creator.add_review(
            candidate["id"],
            reviewer="reviewer1",
            adjudicated_traits=["persecuted_victim"],
            approved=True,
            notes="Good example"
        )
        
        assert candidate["review_state"] == "approved"
        assert candidate["trait_annotations"]["persecuted_victim"] == "present"
        assert len(candidate["review_trail"]) == 1


def test_save_candidates():
    """Test saving candidates with manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        creator = ConspirSyntheticCreator(Path(tmpdir))
        creator.create_candidate("Text 1", ["contradictory"], "science")
        creator.create_candidate("Text 2", ["immune_to_evidence"], "climate")
        
        batch_dir = creator.save_candidates("test_batch")
        assert batch_dir.exists()
        assert (batch_dir / "candidates.jsonl").exists()
        assert (batch_dir / "manifest.json").exists()
        
        with open(batch_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["count"] == 2
```
