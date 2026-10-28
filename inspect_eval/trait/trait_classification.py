"""Trait classification task for CONSPIR."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))

from inspect_ai import Task, task
from inspect_ai.solver import generate, system_message
from inspect_ai.scorer import Score, Target, accuracy, scorer, mean

from trait.trait_dataset import trait_dataset, TRAIT_IDS


TRAIT_PROMPT = """You are an expert at identifying conspiracy theory traits in text.

Classify the given text for the presence of CONSPIR traits. A text may exhibit zero or more traits.

CONSPIR traits:
1. contradictory - Contradictory claims or internal inconsistencies
2. overriding_suspicion - Overriding suspicion of authority/institutions  
3. nefarious_intent - Attribution of nefarious intent
4. something_must_be_wrong - Something must be wrong/there is a hidden truth
5. persecuted_victim - Self or group portrayed as persecuted victim
6. immune_to_evidence - Claims immune to contrary evidence
7. reinterpreting_randomness - Re-interpreting random events as meaningful

Respond with a JSON array of trait IDs that are present. Only include traits that are clearly exhibited.
Example: ["nefarious_intent", "persecuted_victim"]
If no traits are present, respond with [].
"""


@scorer(metrics=[mean()])
def trait_scorer():
    async def score(state, target):
        try:
            completion = state.output.completion
            # Try to extract JSON array
            completion_clean = completion.strip()
            # Handle markdown code blocks
            if "```json" in completion_clean:
                start = completion_clean.find("```json") + 7
                end = completion_clean.find("```", start)
                completion_clean = completion_clean[start:end].strip()
            elif "```" in completion_clean:
                start = completion_clean.find("```") + 3
                end = completion_clean.find("```", start)
                completion_clean = completion_clean[start:end].strip()
            
            predicted = json.loads(completion_clean)
            if not isinstance(predicted, list):
                return Score(value=0.0, explanation="Invalid format")
            
            # Validate trait IDs
            valid_predicted = []
            for t in predicted:
                if isinstance(t, str) and t in TRAIT_IDS:
                    if t not in valid_predicted:  # deduplicate
                        valid_predicted.append(t)
            
            # Get target
            try:
                target_traits = json.loads(str(target))
            except:
                target_traits = []
            
            if not isinstance(target_traits, list):
                target_traits = []
            
            # Compute exact match for now (can be extended to per-trait metrics)
            valid_predicted_set = set(valid_predicted)
            target_set = set(target_traits)
            
            if valid_predicted_set == target_set:
                return Score(value=1.0, explanation=f"Exact match: {valid_predicted}")
            else:
                # Partial credit based on F1
                true_pos = len(valid_predicted_set & target_set)
                if len(valid_predicted_set) == 0 and len(target_set) == 0:
                    return Score(value=1.0, explanation="Both empty")
                prec = true_pos / len(valid_predicted_set) if valid_predicted_set else 0.0
                rec = true_pos / len(target_set) if target_set else 0.0
                if prec + rec == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * prec * rec / (prec + rec)
                return Score(value=f1, explanation=f"Pred: {valid_predicted}, Target: {target_traits}, F1: {f1:.3f}")
        except Exception as e:
            return Score(value=0.0, explanation=f"Error: {str(e)}")
    
    return score


@task
def conspir_trait_classification() -> Task:
    """CONSPIR trait classification task."""
    return Task(
        dataset=trait_dataset(),
        solver=[system_message(TRAIT_PROMPT), generate()],
        scorer=trait_scorer(),
        name="conspir_trait_classification"
    )
