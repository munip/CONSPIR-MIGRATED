"""CONSPIR-based synthetic data creator."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional


TRAIT_IDS = [
    "contradictory",
    "overriding_suspicion",
    "nefarious_intent", 
    "something_must_be_wrong",
    "persecuted_victim",
    "immune_to_evidence",
    "reinterpreting_randomness"
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ConspirSyntheticCreator:
    """Create synthetic CONSPIR trait examples with review gates."""
    
    def __init__(self, output_dir: Path, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.candidates = []
    
    def create_candidate(self,
                        text: str,
                        requested_traits: List[str],
                        domain_id: str,
                        source_ids: Optional[List[str]] = None,
                        model: str = "template",
                        prompt_version: str = "v1",
                        is_counterexample: bool = False) -> Dict[str, Any]:
        """Create a synthetic candidate with full provenance."""
        candidate_id = f"synth-{uuid.uuid4().hex[:12]}"
        
        # Validate traits
        for trait in requested_traits:
            if trait not in TRAIT_IDS:
                raise ValueError(f"Invalid trait: {trait}")
        
        candidate = {
            "schema_version": "2.0",
            "task_type": "trait_classification",
            "id": candidate_id,
            "text": text,
            "domain_id": domain_id,
            "trait_annotations": {t: "unannotated" for t in TRAIT_IDS},
            "source": {
                "dataset": "synthetic",
                "generator": {
                    "model": model,
                    "prompt_version": prompt_version,
                    "seed": self.seed,
                    "timestamp": _utc_now_iso()
                },
                "source_ids": source_ids or [],
                "license": "review_required",
                "requested_traits": requested_traits,
                "is_counterexample": is_counterexample
            },
            "metadata": {
                "review_state": "pending",
                "review_trail": []
            }
        }
        
        self.candidates.append(candidate)
        return candidate
    
    def add_review(self, candidate_id: str, reviewer: str, 
                   adjudicated_traits: List[str],
                   approved: bool,
                   notes: str = ""):
        """Add blind review to candidate."""
        for candidate in self.candidates:
            if candidate["id"] == candidate_id:
                # Update annotations based on review
                if approved:
                    for trait in TRAIT_IDS:
                        if trait in adjudicated_traits:
                            candidate["trait_annotations"][trait] = "present"
                        else:
                            candidate["trait_annotations"][trait] = "absent"
                    candidate["dominant_trait"] = adjudicated_traits[0] if adjudicated_traits else None
                if "metadata" not in candidate:
                    candidate["metadata"] = {}
                candidate["metadata"]["review_state"] = "approved" if approved else "rejected"
                if "review_trail" not in candidate["metadata"]:
                    candidate["metadata"]["review_trail"] = []
                candidate["metadata"]["review_trail"].append({
                    "reviewer": reviewer,
                    "timestamp": _utc_now_iso(),
                    "adjudicated_traits": adjudicated_traits,
                    "approved": approved,
                    "notes": notes
                })
                break
    
    def save_candidates(self, batch_name: Optional[str] = None):
        """Save candidates to batch directory."""
        if batch_name is None:
            batch_name = f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        
        batch_dir = self.output_dir / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all candidates
        candidates_file = batch_dir / "candidates.jsonl"
        with open(candidates_file, 'w', encoding='utf-8') as f:
            for candidate in self.candidates:
                f.write(json.dumps(candidate, ensure_ascii=False) + '\n')
        
        # Save manifest
        manifest = {
            "batch_name": batch_name,
            "created": _utc_now_iso(),
            "seed": self.seed,
            "count": len(self.candidates),
            "review_states": {
                "pending": sum(1 for c in self.candidates if c.get("metadata", {}).get("review_state") == "pending"),
                "approved": sum(1 for c in self.candidates if c.get("metadata", {}).get("review_state") == "approved"),
                "rejected": sum(1 for c in self.candidates if c.get("metadata", {}).get("review_state") == "rejected")
            }
        }
        with open(batch_dir / "manifest.json", 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        return batch_dir
