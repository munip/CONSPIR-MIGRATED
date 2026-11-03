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


def load_trait_items(root: Path = ROOT, dataset_name: str = "conspired",
                     split: str = "test") -> List[Dict[str, Any]]:
    """Load one dataset split; never combine train and test artifacts implicitly."""
    items = []
    if dataset_name == "conspired":
        path = root / "datasets" / "conspired" / f"{split}.jsonl"
        # Backward compatibility for earlier manually normalized artifacts.
        if not path.exists() and split == "test":
            path = root / "datasets" / "conspired" / "normalized.jsonl"
        trait_files = [path]
    elif dataset_name == "loco":
        loco_names = {
            "train": "train_pool_labeled.jsonl",
            "test": "frozen_holdout_labeled.jsonl",
        }
        if split not in loco_names:
            raise ValueError("LOCO split must be train or test")
        path = root / "datasets" / "loco" / loco_names[split]
        if not path.exists() and split == "test":
            path = root / "datasets" / "loco" / "labeled.jsonl"
        trait_files = [path]
    else:
        raise ValueError("dataset_name must be conspired or loco")
    
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
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Trait item {item.get('id', '<unknown>')!r} has empty text")
    context = item.get("context", "")
    
    # Build input
    input_text = text
    if context:
        input_text = f"Context: {context}\n\nText: {text}"
    
    # Get ground truth traits - only use known labels
    trait_annotations = item.get("trait_annotations", {})
    if not isinstance(trait_annotations, dict):
        raise ValueError(f"Trait item {item.get('id', '<unknown>')!r} has invalid trait_annotations")
    target_traits = []
    known_traits = []
    for trait_id in TRAIT_IDS:
        # Upstream datasets such as ConspirED do not necessarily include all seven
        # traits. A missing field is unknown, never an implicit negative.
        ann = trait_annotations.get(trait_id, "unannotated")
        if ann == "unannotated":
            continue  # Do not treat unannotated as negative
        if ann not in ("present", "absent"):
            raise ValueError(
                f"Trait item {item.get('id', '<unknown>')!r} has invalid annotation "
                f"for {trait_id!r}: {ann!r}"
            )
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


def trait_dataset(root: Path = ROOT, dataset_name: str = "conspired",
                  split: str = "test") -> MemoryDataset:
    """Create dataset for trait classification."""
    items = load_trait_items(root, dataset_name=dataset_name, split=split)
    samples = []
    for item in items:
        sample = trait_to_sample(item)
        # Rows with no gold labels cannot be evaluated; retain them in source
        # artifacts, but do not count them as empty-label examples.
        if sample.metadata.get("known_traits"):
            samples.append(sample)
    if not samples:
        raise ValueError(
            f"No labeled CONSPIR trait rows found for dataset={dataset_name!r}, split={split!r}. "
            "Import and annotate data first; fully unannotated rows are not evaluation examples."
        )
    return MemoryDataset(samples, name=f"conspir_trait_{dataset_name}_{split}")
