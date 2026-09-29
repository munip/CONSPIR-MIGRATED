"""A quick check of a judge model against the labelled cases in eval/judge_cases*.jsonl.

    inspect eval inspect_eval/judge_cases.py --model <judge>

The model here IS the judge: each sample is the exact prompt ``conspiracy_judge`` would
send for that case (single-turn, or with the case's transcript), and the score is
whether the label ``parse_label`` reads from its reply matches ``expected_label``.
Run it for whichever judge a benchmark run uses, so the benchmark's numbers come with
a figure for how far that judge can be trusted. It is a smoke check over 26 cases, not
the judge-reliability study in eval/judge_reliability/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.model import GenerateConfig  # noqa: E402
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr  # noqa: E402
from inspect_ai.solver import TaskState, generate  # noqa: E402

from scorer import INVALID, build_judge_prompt, judge_reply_text, parse_label  # noqa: E402

CASE_FILES = ("judge_cases.jsonl", "judge_cases_multiturn.jsonl")


def judge_case_dataset(root: Path = HERE.parent) -> MemoryDataset:
    samples = []
    for name in CASE_FILES:
        for line in (root / "eval" / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            prompt = build_judge_prompt(case["item"], case["model_answer"], case.get("transcript") or None)
            samples.append(Sample(
                input=prompt,
                target=case["expected_label"],
                id=case["case_id"],
                metadata={"multiturn": bool(case.get("transcript")), "purpose": case.get("purpose")},
            ))
    return MemoryDataset(samples, name="judge_cases")


@scorer(metrics=[accuracy(), stderr()])
def matches_expected_label():
    async def score(state: TaskState, target: Target) -> Score:
        label = parse_label(judge_reply_text(state.output.completion)) or INVALID
        return Score(value=1 if label == target.text else 0, answer=label,
                     explanation=f"expected {target.text}, judged {label}")

    return score


@task
def judge_cases() -> Task:
    return Task(
        dataset=judge_case_dataset(),
        solver=generate(),
        scorer=matches_expected_label(),
        config=GenerateConfig(temperature=0),  # as conspiracy_judge runs it
    )
