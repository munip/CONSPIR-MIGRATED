"""ConspiracyBench as an Inspect task. A pilot that runs alongside eval/, not instead of it (#42).

The judge prompt, the label parsing and the outcome arithmetic are imported from the
existing code rather than reimplemented, so this module only adds Inspect's dataset,
solver and scorer plumbing around them:

- the judge prompt comes from ``eval/judge_prompt.py``;
- judge output is parsed with the same functions as ``eval/judge_reliability/run_baseline.py``;
- rates come from ``eval/score_outcomes.py``.

Run it with a model under test and an explicit judge. The judge is a required model role,
so a run can never silently fall back to grading a model with itself:

    inspect eval inspect_eval/conspiracybench.py --model <model> --model-role grader=<judge>
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import SampleScore, Score, Target, metric, scorer
from inspect_ai.solver import Generate, TaskState, solver

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from judge_prompt import build_judge_prompt  # noqa: E402
from score_outcomes import VALID_OUTCOMES, summarize_outcomes, summarize_trajectories  # noqa: E402


def _load_label_parser():
    path = ROOT / "eval" / "judge_reliability" / "run_baseline.py"
    spec = importlib.util.spec_from_file_location("_judge_label_parser", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_parser = _load_label_parser()

# One judge label per turn, joined into a single string. The score value has to be a
# scalar: Inspect's `collect` reducer only accepts scalars, and it is the one reducer that
# keeps every epoch's result. A dict value would be reduced key by key, mixing turns from
# different epochs, and Score.metadata keeps only the first epoch.
TURN_SEPARATOR = ">"
INVALID = "invalid"


# --------------------------------------------------------------------------- dataset


def load_items(root: Path = ROOT, include_pending: bool = True):
    """Yield (path, line number, item) for the corpus and, optionally, pending submissions."""
    paths = [root / "questions" / "corpus.jsonl"]
    if include_pending:
        paths += sorted((root / "questions" / "pending").glob("*.jsonl"))
    for path in paths:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                yield path, lineno, json.loads(line)


def item_to_sample(item: dict, sample_id: str) -> Sample:
    return Sample(
        input=item["question"],
        target=item["ground_truth_answer"],
        id=sample_id,
        metadata={"item": item},
    )


def conspiracybench_dataset(root: Path = ROOT, include_pending: bool = True) -> MemoryDataset:
    """Every item in the corpus, plus pending submissions unless excluded.

    Pending items all share the placeholder id ``cb-pending``, so they are given a sample
    id from their file and line instead, e.g. ``pending/wtc7-collapse:1``.
    """
    samples = []
    for path, lineno, item in load_items(root, include_pending):
        sample_id = item["id"] if item["id"] != "cb-pending" else f"pending/{path.stem}:{lineno}"
        samples.append(item_to_sample(item, sample_id))
    return MemoryDataset(samples, name="conspiracybench")


# ---------------------------------------------------------------------------- solver


@solver
def pressure_turns():
    """Answer the question, then each of the item's ``pressure_turns`` in order.

    The model's own replies stay in context between turns. Items without pressure turns
    get a single generation, so single-turn items run exactly as a plain question would.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)
        for turn in state.metadata["item"].get("pressure_turns") or []:
            state.messages.append(ChatMessageUser(content=turn))
            state = await generate(state)
        return state

    return solve


# ---------------------------------------------------------------------------- scorer


def _graded_turns(messages):
    """Pair each model answer with the conversation that preceded it."""
    history, turns = [], []
    for message in messages:
        if message.role not in ("user", "assistant"):
            continue
        if message.role == "assistant":
            turns.append((list(history), message.text))
        history.append({"role": message.role, "content": message.text})
    return turns


@scorer(metrics=[])
def conspiracy_judge(grader_role: str = "grader"):
    """Label every turn hold, diverge or abstain with the ConspiracyBench judge prompt.

    The opening answer is graded with the single-turn prompt (no transcript), which is the
    prompt every existing item has always been graded with. Each later answer is graded with
    the preceding exchange as context, as ``eval/judge_prompt.py`` does for pressure turns.
    The judge runs at temperature 0 so its variance stays separate from the model's (#38).
    """

    async def score(state: TaskState, target: Target) -> Score:
        item = state.metadata["item"]
        judge = get_model(role=grader_role, required=True)
        labels, raw_outputs, strict_labels = [], [], []
        answer = ""
        for index, (history, answer) in enumerate(_graded_turns(state.messages)):
            prompt = build_judge_prompt(item, answer, history if index else None)
            output = await judge.generate(prompt, config=GenerateConfig(temperature=0))
            raw = output.completion
            raw_outputs.append(raw)
            strict_labels.append(_parser.strict_parse(raw))
            labels.append(_parser.permissive_parse(raw) or INVALID)
        return Score(
            value=TURN_SEPARATOR.join(labels),
            answer=answer,
            explanation="\n---\n".join(raw_outputs),
            metadata={"turn_labels": labels, "strict_turn_labels": strict_labels},
        )

    return score


# --------------------------------------------------------------------------- metrics


def _trajectories(scores: list[SampleScore]) -> list[list[str]]:
    """Every graded trajectory, one per sample per epoch.

    After the `collect` reducer a sample's value is a list with one entry per epoch; a
    single unreduced score is a plain string.
    """
    trajectories = []
    for sample_score in scores:
        value = sample_score.score.value
        for entry in value if isinstance(value, list) else [value]:
            trajectories.append(str(entry).split(TURN_SEPARATOR))
    return trajectories


def _valid(trajectories):
    return [t for t in trajectories if all(label in VALID_OUTCOMES for label in t)]


def _nan_if_none(value):
    return math.nan if value is None else value


@metric
def divergence_rate():
    """Divergence over answered opening turns, as defined in SCHEMA.md."""

    def compute(scores: list[SampleScore]) -> float:
        opening = [t[0] for t in _valid(_trajectories(scores))]
        return _nan_if_none(summarize_outcomes(opening)["divergence_rate"]) if opening else math.nan

    return compute


@metric
def abstention_rate():
    """Abstention over all opening turns."""

    def compute(scores: list[SampleScore]) -> float:
        opening = [t[0] for t in _valid(_trajectories(scores))]
        return _nan_if_none(summarize_outcomes(opening)["abstention_rate"]) if opening else math.nan

    return compute


def _pressured(scores):
    """Multi-turn trajectories only: a single-turn item cannot concede or retreat."""
    return [t for t in _valid(_trajectories(scores)) if len(t) > 1]


@metric
def concession_rate():
    """hold -> diverge, over multi-turn trajectories that open with hold."""

    def compute(scores: list[SampleScore]) -> float:
        pressured = _pressured(scores)
        return _nan_if_none(summarize_trajectories(pressured)["concession_rate"]) if pressured else math.nan

    return compute


@metric
def retreat_rate():
    """hold -> abstain, over multi-turn trajectories that open with hold."""

    def compute(scores: list[SampleScore]) -> float:
        pressured = _pressured(scores)
        return _nan_if_none(summarize_trajectories(pressured)["retreat_rate"]) if pressured else math.nan

    return compute


@metric
def invalid_rate():
    """Share of trajectories where the judge output could not be parsed on some turn.

    These are excluded from every other rate rather than counted as any label.
    """

    def compute(scores: list[SampleScore]) -> float:
        trajectories = _trajectories(scores)
        if not trajectories:
            return math.nan
        return sum(1 for t in trajectories if INVALID in t) / len(trajectories)

    return compute


METRICS = [divergence_rate(), abstention_rate(), concession_rate(), retreat_rate(), invalid_rate()]


# ------------------------------------------------------------------------------ task


def build_task(dataset, epochs: int = 1, grader_role: str = "grader") -> Task:
    """The ConspiracyBench task over any dataset. ``conspiracybench`` below uses the repo's."""
    return Task(
        dataset=dataset,
        solver=pressure_turns(),
        scorer=conspiracy_judge(grader_role),
        metrics=METRICS,
        epochs=Epochs(epochs, "collect"),
    )


@task
def conspiracybench(include_pending: bool = True, epochs: int = 1, grader_role: str = "grader") -> Task:
    return build_task(conspiracybench_dataset(include_pending=include_pending), epochs, grader_role)
