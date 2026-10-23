#!/usr/bin/env python3
"""Export every gradable turn of an Inspect log as units for the decision judges.

    python inspect_eval/export_units.py logs/<file>.eval --out units.jsonl

The deployed judges (pplx-decider, Clef) are decision models, not chat models, so they can't be
Inspect's grader role. Run the solver with ``--no-score`` (or keep the chat judge's scores), export
the turns with this, and grade them with ``decision_judges/`` (``grade_units.py``, on Modal via
``modal_grade.py``). Units have the validity set's shape (``case_id``, ``item``, ``model_answer``,
``transcript``, ``turn``), so the judges get exactly the request they were validated with.

``transcript`` is the whole conversation before the reply for a later turn, and empty for the
opening question, as in the Inspect scorer. Empty and cut-off replies are not graded, for
the same reasons the scorer doesn't grade them: they are written with ``skip`` set to the reason,
so the report can still rebuild every trajectory turn by turn. ``--grade-truncated`` grades
cut-off ones.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cb_budget import EMPTY_REPLY, MODEL_CONTEXT_OVERFLOW, OVERFLOW_TURN, TRUNCATED_REPLY, TRUNCATED_TURNS  # noqa: E402
from report import BREAKDOWNS  # noqa: E402
from scorer.cb_scorer import _graded_turns  # noqa: E402


def units_from_log(log, grade_truncated: bool = False) -> tuple[list[dict], Counter]:
    """Every turn of the log as a unit, and a count of the turns not to grade (``skip``), by reason."""
    units, skipped = [], Counter()
    for sample in log.samples or []:
        metadata = sample.metadata or {}
        truncated = set(metadata.get(TRUNCATED_TURNS) or [])
        for index, (history, answer) in enumerate(_graded_turns(sample.messages)):
            skip = None
            if not answer.strip():
                skip = EMPTY_REPLY
            elif index in truncated and not grade_truncated:
                skip = TRUNCATED_REPLY
            if skip:
                skipped[skip] += 1
            units.append({
                "case_id": f"{sample.id}#e{sample.epoch}#t{index}",
                "sample_id": str(sample.id),
                "epoch": sample.epoch,
                "turn": index,
                "item": metadata["item"],
                "model_answer": answer,
                "transcript": history if index else [],
                "skip": skip,
                "breakdown": {field: metadata.get(field) for field in BREAKDOWNS},
            })
        overflow = metadata.get(OVERFLOW_TURN)
        if overflow is not None:
            # The solver never asked this turn: even the latest message alone didn't fit.
            skipped[MODEL_CONTEXT_OVERFLOW] += 1
            units.append({"case_id": f"{sample.id}#e{sample.epoch}#t{overflow}", "sample_id": str(sample.id),
                          "epoch": sample.epoch, "turn": overflow,
                          "item": metadata["item"], "model_answer": "", "transcript": [],
                          "skip": MODEL_CONTEXT_OVERFLOW,
                          "breakdown": {field: metadata.get(field) for field in BREAKDOWNS}})
    return units, skipped


def main(argv=None):
    from inspect_ai.log import read_eval_log

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grade-truncated", action="store_true")
    args = parser.parse_args(argv)
    log = read_eval_log(str(args.log))
    units, skipped = units_from_log(log, args.grade_truncated)
    with args.out.open("w", encoding="utf-8") as f:
        for unit in units:
            f.write(json.dumps(unit, ensure_ascii=False) + "\n")
    print(f"{sum(not u['skip'] for u in units)} units to grade from {log.eval.model} -> {args.out}"
          + (f"; left out: {dict(skipped)}" if skipped else ""))


if __name__ == "__main__":
    main()
