"""The ConspiracyBench solver: the opening question, then the item's pressure turns."""
from __future__ import annotations

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, TaskState, solver


@solver
def pressure_turns(pressure_ids: list[str] | None = None):
    """Answer the question, then each of the item's ``pressure_turns`` in order.

    The model's own replies stay in context between turns. Items without pressure turns
    get a single generation, so single-turn items run exactly as a plain question would.

    ``pressure_ids`` applies the repeated-sampling protocol in SCHEMA.md: every epoch
    samples the opening turn, but the pressure sequence plays only in epoch 1, except for
    the listed sample ids, which play it in every epoch. ``None`` plays it in every epoch
    for every item.
    """
    repeat = None if pressure_ids is None else set(pressure_ids)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)
        if repeat is not None and state.epoch != 1 and str(state.sample_id) not in repeat:
            return state
        for turn in state.metadata["item"].get("pressure_turns") or []:
            state.messages.append(ChatMessageUser(content=turn))
            state = await generate(state)
        return state

    return solve
