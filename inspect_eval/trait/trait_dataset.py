"""Dataset loader for CONSPIR trait classification."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from inspect_ai.dataset import MemoryDataset, Sample

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIT_IDS = [
    "contradictory",
    "overriding_suspicion", 
    "nefarious_intent",
    "something_must_be_wrong",
    "persecuted_victim",
    "immune_to_evidence",
    "reinterpreting_randomness"
]


def load_trait_items(root: Path = ROOT) -> List[Dict[str, Any]]:
    """Load trait classification items."""
    items = []
    trait_files = [
        root / "datasets" / "conspired" / "normalized.jsonl",
        root / "datasets" / "loco" / "labeled.jsonl",
        root / "datasets" / "trait" / "items.jsonl"
    ]
    
    for trait_file in trait_files:
        if trait_file.exists():
            with open(trait_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        items.append(item)
    
    return items


def trait_to_sample(item: Dict[str, Any]) -> Sample:
    """Convert trait item to Inspect Sample."""
    text = item.get("text", "")
    context = item.get("context", "")
    
    # Build input
    input_text = text
    if context:
        input_text = f"Context: {context}\n\nText: {text}"
    
    # Get ground truth traits - only use known labels
    trait_annotations = item.get("trait_annotations", {})
    target_traits = []
    known_traits = []
    for trait_id in TRAIT_IDS:
        ann = trait_annotations.get(trait_id)
        if ann == "unannotated":
            continue  # Do not treat unannotated as negative
        known_traits.append(trait_id)
        if ann == "present":
            target_traits.append(trait_id)
    
    return Sample(
        input=input_text,
        target=json.dumps(sorted(target_traits)),
        id=item.get("id", "unknown"),
        metadata={
            "item": item,
            "domain_id": item.get("domain_id"),
            "source": item.get("source", {}),
            "trait_annotations": trait_annotations,
            "known_traits": known_traits
        }
    )


def trait_dataset(root: Path = ROOT) -> MemoryDataset:
    """Create dataset for trait classification."""
    items = load_trait_items(root)
    samples = [trait_to_sample(item) for item in items]
    return MemoryDataset(samples, name="conspir_trait")
