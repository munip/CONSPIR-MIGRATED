"""The ConspiracyBench solver: the opening question, then the item's pressure turns."""
from __future__ import annotations

from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.solver import Generate, TaskState, solver

from cb_budget import (
    CUT_OFF_STOP_REASONS,
    DEFAULT_REPLY_TOKENS,
    OVERFLOW_TURN,
    STOP_REASONS,
    TRUNCATED_TURNS,
    TURNS_PLANNED,
    fits,
)


@solver
def pressure_turns(
    pressure_ids: list[str] | None = None,
    context_tokens: int | None = None,
    max_tokens: int | None = None,
):
    """Answer the question, then each of the item's ``pressure_turns`` in order.

    The model's own replies stay in context between turns. Items without pressure turns
    get a single generation, so single-turn items run exactly as a plain question would.

    ``pressure_ids`` applies the repeated-sampling protocol in SCHEMA.md: every epoch
    samples the opening turn, but the pressure sequence plays only in epoch 1, except for
    the listed sample ids, which play it in every epoch. ``None`` plays it in every epoch
    for every item.

    ``context_tokens`` is the token budget for the conversation (see ``cb_budget``). A turn
    whose prompt plus ``max_tokens`` of reply would not fit is not sent, since a provider
    may silently drop the start of it; the sequence stops there and the turn is recorded
    in the sample's metadata for the scorer. ``None`` turns the check off.

    Why each reply ended is recorded in the sample's metadata, one Inspect stop reason per turn,
    as ``stop_reasons``. The turns that were cut off (they hit ``max_tokens`` or the model's own
    limit) are recorded as ``truncated_turns``, worked out from that list.
    """
    repeat = None if pressure_ids is None else set(pressure_ids)
    reply_tokens = max_tokens or DEFAULT_REPLY_TOKENS

    async def generate_if_it_fits(state: TaskState, generate: Generate) -> tuple[TaskState, bool]:
        if not await fits(get_model(), state.messages, reply_tokens, context_tokens):
            state.metadata[OVERFLOW_TURN] = sum(1 for m in state.messages if m.role == "assistant")
            return state, False
        state = await generate(state)
        reasons = state.metadata.setdefault(STOP_REASONS, [])
        reasons.append(state.output.stop_reason)
        # A reply that ran out of room before the model finished it. Derived from the list, so the
        # scorer can tell a complete answer from one that stops mid-sentence, and the two agree.
        cut_off = [turn for turn, reason in enumerate(reasons) if reason in CUT_OFF_STOP_REASONS]
        if cut_off:
            state.metadata[TRUNCATED_TURNS] = cut_off
        return state, True

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Checked before any model call, so a malformed item doesn't spend one first.
        turns = state.metadata["item"].get("pressure_turns")
        if turns is not None:
            if not isinstance(turns, list):
                raise TypeError("pressure_turns must be a list")
            for turn in turns:
                if not isinstance(turn, str):
                    raise TypeError("each turn in pressure_turns must be a string")
        # Recorded before the first call, so a sample that errors partway still says how far it
        # should have gone.
        plays_pressure = repeat is None or state.epoch == 1 or str(state.sample_id) in repeat
        state.metadata[TURNS_PLANNED] = 1 + len(turns or []) if plays_pressure else 1
        state, sent = await generate_if_it_fits(state, generate)
        if not sent:
            return state
        if not plays_pressure:
            return state
        for turn in turns or []:
            state.messages.append(ChatMessageUser(content=turn))
            state, sent = await generate_if_it_fits(state, generate)
            if not sent:
                break
        return state

    return solve
