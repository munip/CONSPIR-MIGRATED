"""ConspiracyBench as an Inspect task: dataset, solver, judge scorer and metrics.

The judge prompt, the label parsing and the outcome arithmetic are imported from the
existing code rather than reimplemented, so this folder only adds Inspect's dataset,
solver and scorer plumbing around them:

- the judge prompt comes from ``eval/judge_prompt.py``;
- judge output is parsed with the same functions as ``completed_experiments/judge_reliability/run_baseline.py``;
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

from cb_budget import DEFAULT_CONTEXT_TOKENS  # noqa: E402
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

# How long one failing model call is retried before the sample errors. Inspect retries for
# ever unless told otherwise, so a dead endpoint would stall a run without failing it. Its
# backoff starts at 3 s and doubles, so 10 retries tolerate about 50 minutes of outage, and
# no new attempt starts once the timeout has passed. Neither interrupts an attempt that is
# still waiting for a reply: only attempt_timeout does (README, "A dead endpoint"). Both
# apply to the judge as well.
DEFAULT_MAX_RETRIES = 10
DEFAULT_TIMEOUT = 3600


def _optional_int(value):
    """An int, or None for ``none`` / ``null``.

    ``-T name=none`` reaches a task as the string "none", not None. Left alone it would end up in
    the config (the log then cannot be read back) or fail later in a comparison. Anything that
    is not a number or none raises here, before the run starts. That includes an empty value
    (``-T max_retries=``), which would otherwise quietly remove a limit.
    """
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("none", "null"):
            return None
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"expected a whole number or none, got {value!r}") from None
    return value


def build_task(
    dataset,
    epochs: int = 1,
    grader_role: str = "grader",
    pressure_ids: list[str] | None = None,
    system_prompt: str | None = None,
    config: GenerateConfig | None = None,
    context_tokens: int | None = DEFAULT_CONTEXT_TOKENS,
    judge_context_tokens: int | None = None,
    max_retries: int | None = DEFAULT_MAX_RETRIES,
    timeout: int | None = DEFAULT_TIMEOUT,
    grade_truncated: bool = False,
) -> Task:
    """The ConspiracyBench task over any dataset. ``conspiracybench`` below uses the repo's.

    ``pressure_ids=None`` plays every pressure sequence in every epoch; a list limits the
    repeats after epoch 1 to those samples (see ``pressure_turns``).

    ``context_tokens`` is the token budget every prompt must fit in, for the model under
    test and, unless ``judge_context_tokens`` is given, for the judge too (see
    ``cb_budget``). A turn that would not fit is not sent and is scored invalid, rather
    than risk a provider silently truncating it. ``context_tokens=None`` turns it off.

    ``max_retries`` and ``timeout`` bound how long one failing model call is retried, for
    the model under test and for the judge. Inspect's own default is to retry for ever, so
    a dead endpoint would stall a run without failing it. A value already set in ``config``
    wins, ``None`` means no limit, and ``--max-retries`` / ``--timeout`` override both.

    A reply that was cut off before the model finished it is scored invalid
    (``truncated_reply``) rather than graded as if complete. ``grade_truncated=True`` sends
    such replies to the judge anyway; they are counted in the report either way.
    """
    config = config or GenerateConfig()
    config = config.model_copy(update={
        name: value
        for name, value in (("max_retries", max_retries), ("timeout", timeout))
        if getattr(config, name) is None
    })
    if judge_context_tokens is None:
        judge_context_tokens = context_tokens
    solver = [pressure_turns(pressure_ids, context_tokens=context_tokens, max_tokens=config.max_tokens)]
    if system_prompt:
        # system_message() fills {name} placeholders from the sample's metadata, so a prompt
        # containing JSON would be mangled and {item} would hand the model the ground truth.
        # Doubling the braces makes the prompt go out exactly as written.
        solver.insert(0, system_message(system_prompt.replace("{", "{{").replace("}", "}}")))
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=conspiracy_judge(
            grader_role, context_tokens=judge_context_tokens, grade_truncated=grade_truncated
        ),
        metrics=METRICS,
        epochs=Epochs(epochs, "collect"),
        config=config,
    )


@task
def conspiracybench(
    include_pending: bool = False,
    epochs: int = 1,
    pressure_subset_per_cell: int = 1,
    subset_seed: int = 0,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = 2048,
    context_tokens: int | None = DEFAULT_CONTEXT_TOKENS,
    judge_context_tokens: int | None = None,
    max_retries: int | None = DEFAULT_MAX_RETRIES,
    timeout: int | None = DEFAULT_TIMEOUT,
    grade_truncated: bool = False,
    grader_role: str = "grader",
) -> Task:
    """Every item, sampled ``epochs`` times on the opening turn.

    Pressure sequences run in epoch 1 for every multi-turn item and in every epoch for a
    stratified subset: up to ``pressure_subset_per_cell`` items per category x difficulty
    band, drawn with ``subset_seed``. The subset's ids are recorded in the log's task
    arguments via the solver. ``temperature=None`` keeps the model's own default, which
    is what SCHEMA.md asks for; ``system_prompt=None`` sends no system message.

    ``context_tokens`` must not exceed the context the models are actually loaded with
    (``num_ctx`` in the Ollama Modelfile). Pass ``judge_context_tokens`` when the judge's
    context differs from the model under test's.

    ``max_retries`` and ``timeout`` are how long one failing model call is retried, for the
    model and the judge (see ``DEFAULT_MAX_RETRIES``).
    """
    context_tokens, judge_context_tokens = _optional_int(context_tokens), _optional_int(judge_context_tokens)
    max_retries, timeout = _optional_int(max_retries), _optional_int(timeout)
    dataset = conspiracybench_dataset(include_pending=include_pending)
    pressure_ids = stratified_subset(dataset, pressure_subset_per_cell, subset_seed)
    return build_task(
        dataset,
        epochs=epochs,
        grader_role=grader_role,
        pressure_ids=pressure_ids,
        system_prompt=system_prompt,
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
        context_tokens=context_tokens,
        judge_context_tokens=judge_context_tokens,
        max_retries=max_retries,
        timeout=timeout,
        grade_truncated=grade_truncated,
    )
