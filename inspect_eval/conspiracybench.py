"""ConspiracyBench as an Inspect task: dataset, solver, judge scorer and metrics.

The judge prompt, the label parsing and the outcome arithmetic are imported from the
existing code rather than reimplemented, so this folder only adds Inspect's dataset,
solver and scorer plumbing around them:

- the judge prompt comes from ``eval/judge_prompt.py``;
- judge output is parsed with the same functions as ``eval/judge_reliability/run_baseline.py``;
- rates come from ``eval/score_outcomes.py``.

The pieces live in ``cb_dataset.py``, ``solver/``, ``scorer/`` and
``cb_metrics.py``; this file is the task entry point. Run it with a model under test and
an explicit judge. The judge is a required model role, so a run can never silently fall
back to grading a model with itself:

    inspect eval inspect_eval/conspiracybench.py --model <model> --model-role grader=<judge>
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from inspect_ai import Epochs, Task, task  # noqa: E402
from inspect_ai.model import GenerateConfig  # noqa: E402
from inspect_ai.solver import system_message  # noqa: E402

from cb_dataset import (  # noqa: E402, F401  (re-exported for the tests and README)
    ROOT,
    conspiracybench_dataset,
    item_to_sample,
    load_items,
    stratified_subset,
)
from cb_metrics import METRICS  # noqa: E402
from scorer import INVALID, TURN_SEPARATOR, conspiracy_judge  # noqa: E402, F401
from solver import pressure_turns  # noqa: E402

# k for the documented full run (SCHEMA.md, "Repeated sampling"). The task defaults to
# one epoch so a quick run stays quick; pass -T epochs=5 for a benchmark run.
BENCHMARK_EPOCHS = 5


def build_task(
    dataset,
    epochs: int = 1,
    grader_role: str = "grader",
    pressure_ids: list[str] | None = None,
    system_prompt: str | None = None,
    config: GenerateConfig | None = None,
) -> Task:
    """The ConspiracyBench task over any dataset. ``conspiracybench`` below uses the repo's.

    ``pressure_ids=None`` plays every pressure sequence in every epoch; a list limits the
    repeats after epoch 1 to those samples (see ``pressure_turns``).
    """
    solver = [pressure_turns(pressure_ids)]
    if system_prompt:
        solver.insert(0, system_message(system_prompt))
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=conspiracy_judge(grader_role),
        metrics=METRICS,
        epochs=Epochs(epochs, "collect"),
        config=config or GenerateConfig(),
    )


@task
def conspiracybench(
    include_pending: bool = True,
    epochs: int = 1,
    pressure_subset_per_cell: int = 1,
    subset_seed: int = 0,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = 2048,
    grader_role: str = "grader",
) -> Task:
    """Every item, sampled ``epochs`` times on the opening turn.

    Pressure sequences run in epoch 1 for every multi-turn item and in every epoch for a
    stratified subset: up to ``pressure_subset_per_cell`` items per category x difficulty
    band, drawn with ``subset_seed``. The subset's ids are recorded in the log's task
    arguments via the solver. ``temperature=None`` keeps the model's own default, which
    is what SCHEMA.md asks for; ``system_prompt=None`` sends no system message.
    """
    dataset = conspiracybench_dataset(include_pending=include_pending)
    pressure_ids = stratified_subset(dataset, pressure_subset_per_cell, subset_seed)
    return build_task(
        dataset,
        epochs=epochs,
        grader_role=grader_role,
        pressure_ids=pressure_ids,
        system_prompt=system_prompt,
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
    )
