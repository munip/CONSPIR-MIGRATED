"""Compare response-robustness judge outputs with independent human labels."""
from __future__ import annotations

from collections import Counter
from typing import Iterable


JUDGMENT_FIELDS = (
    "factual_alignment",
    "conspiracy_endorsement",
    "abstention",
)


def _key(row: dict) -> tuple[str, int, int]:
    sample_id = row.get("sample_id")
    turn_index = row.get("turn_index")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Each judgment needs a non-empty sample_id")
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
        raise ValueError(f"{sample_id}: turn_index must be a non-negative integer")
    epoch = row.get("epoch", 1)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError(f"{sample_id}: epoch must be a positive integer")
    return sample_id, epoch, turn_index


def _cohen_kappa(pairs: list[tuple[object, object]]) -> float | None:
    if not pairs:
        return None
    labels = {value for pair in pairs for value in pair}
    observed = sum(left == right for left, right in pairs) / len(pairs)
    expected = sum(
        (sum(left == label for left, _ in pairs) / len(pairs))
        * (sum(right == label for _, right in pairs) / len(pairs))
        for label in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else (1.0 if observed == 1 else 0.0)


def validate_judgments(human_rows: Iterable[dict], judge_rows: Iterable[dict]) -> dict:
    """Return exact agreement and Cohen's kappa for each independent rubric axis."""
    human, judge = {}, {}
    invalid_judge_keys = set()
    for destination, rows, label in (
        (human, human_rows, "human"),
        (judge, judge_rows, "judge"),
    ):
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{label} judgments must be JSON objects")
            key = _key(row)
            if key in destination:
                raise ValueError(f"Duplicate {label} judgment for {key}")
            if label == "judge" and row.get("invalid") is True:
                invalid_judge_keys.add(key)
                destination[key] = row
                continue
            missing = set(JUDGMENT_FIELDS) - row.keys()
            if missing:
                raise ValueError(f"{label} judgment {key} is missing {sorted(missing)}")
            if (not isinstance(row["factual_alignment"], str)
                    or row["factual_alignment"] not in {"matches", "contradicts", "does_not_answer"}):
                raise ValueError(f"{label} judgment {key} has invalid factual_alignment")
            if (not isinstance(row["conspiracy_endorsement"], str)
                    or row["conspiracy_endorsement"] not in {"endorses", "does_not_endorse", "ambiguous"}):
                raise ValueError(f"{label} judgment {key} has invalid conspiracy_endorsement")
            if not isinstance(row["abstention"], bool):
                raise ValueError(f"{label} judgment {key} has invalid abstention")
            destination[key] = row

    valid_judge = {key: row for key, row in judge.items() if key not in invalid_judge_keys}
    matched_keys = sorted(human.keys() & valid_judge.keys())
    result = {
        "reference_units": len(human),
        "judge_units": len(judge),
        "valid_judge_units": len(valid_judge),
        "invalid_judge_units": len(invalid_judge_keys),
        "compared_units": len(matched_keys),
        "human_units_without_valid_judge_output": len(human.keys() - valid_judge.keys()),
        "valid_judge_units_without_human_reference": len(valid_judge.keys() - human.keys()),
        "axes": {},
    }
    for field in JUDGMENT_FIELDS:
        pairs = [(human[key][field], valid_judge[key][field]) for key in matched_keys]
        result["axes"][field] = {
            "exact_agreement": sum(left == right for left, right in pairs) / len(pairs) if pairs else None,
            "cohen_kappa": _cohen_kappa(pairs),
            "confusion": {
                f"{reference} -> {predicted}": count
                for (reference, predicted), count in sorted(Counter(pairs).items(), key=str)
            },
        }
    return result
