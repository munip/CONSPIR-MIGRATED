"""Metrics for CONSPIR trait evaluation."""
from __future__ import annotations

from typing import Dict, List, Set, Any
from collections import defaultdict


TRAIT_IDS = [
    "contradictory", "overriding_suspicion", "nefarious_intent",
    "something_must_be_wrong", "persecuted_victim", 
    "immune_to_evidence", "reinterpreting_randomness"
]


def compute_trait_metrics(predictions: Dict[str, List[str]], 
                         references: Dict[str, Dict[str, str]],
                         known_label_mask: Dict[str, Set[str]] = None,
                         invalid_outputs: int = 0,
                         total_outputs: int = None) -> Dict[str, Any]:
    """Compute per-trait precision, recall, F1 with known-label masking."""
    if known_label_mask is None:
        known_label_mask = {pid: set(TRAIT_IDS) for pid in predictions}
    
    if total_outputs is None:
        total_outputs = len(predictions)
    
    results = {
        "per_trait": {},
        "macro": {"precision": 0, "recall": 0, "f1": 0},
        "micro": {"precision": 0, "recall": 0, "f1": 0},
        "exact_match": 0,
        "invalid_output_rate": invalid_outputs / total_outputs if total_outputs > 0 else 0.0
    }
    
    # Per-trait metrics
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    for trait in TRAIT_IDS:
        tp = fp = fn = tn = 0
        for pid in predictions:
            pred_has = trait in predictions[pid]
            ref_ann = references.get(pid, {}).get(trait, "unannotated")
            
            # Skip if not in known label mask
            if trait not in known_label_mask.get(pid, set(TRAIT_IDS)):
                continue
            if ref_ann == "unannotated":
                continue  # Don't score unannotated labels as negatives
            
            ref_has = (ref_ann == "present")
            
            if pred_has and ref_has:
                tp += 1
            elif pred_has and not ref_has:
                fp += 1
            elif not pred_has and ref_has:
                fn += 1
            else:
                tn += 1
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        results["per_trait"][trait] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": tp + fn,
            "tp": tp, "fp": fp, "fn": fn
        }
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
    
    # Macro averages
    if results["per_trait"]:
        results["macro"]["precision"] = sum(v["precision"] for v in results["per_trait"].values()) / len(results["per_trait"])
        results["macro"]["recall"] = sum(v["recall"] for v in results["per_trait"].values()) / len(results["per_trait"])
        results["macro"]["f1"] = sum(v["f1"] for v in results["per_trait"].values()) / len(results["per_trait"])
    
    # Micro averages
    results["micro"]["precision"] = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    results["micro"]["recall"] = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    results["micro"]["f1"] = 2 * results["micro"]["precision"] * results["micro"]["recall"] / \
                              (results["micro"]["precision"] + results["micro"]["recall"]) \
                              if (results["micro"]["precision"] + results["micro"]["recall"]) > 0 else 0.0
    
    # Exact match
    exact_matches = 0
    for pid in predictions:
        pred_set = set(predictions[pid])
        # Get target set (only present among known)
        ref = references.get(pid, {})
        target_set = set()
        known = known_label_mask.get(pid, set(TRAIT_IDS))
        for t in known:
            if ref.get(t) == "present":
                target_set.add(t)
        if pred_set == target_set:
            exact_matches += 1
    results["exact_match"] = exact_matches / len(predictions) if predictions else 0.0
    
    return results
```
