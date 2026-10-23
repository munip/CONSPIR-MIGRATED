"""The ConspiracyBench solver: the opening question, then the item's pressure turns."""
from __future__ import annotations

from inspect_ai.model import ChatMessageUser, ContentReasoning, ContentText, get_model
from inspect_ai.solver import Generate, TaskState, solver

from cb_budget import (
    CONTEXT_CLIPPED,
    CONTEXT_OVERFLOW_MODES,
    CUT_OFF_STOP_REASONS,
    DEFAULT_CONTEXT_OVERFLOW,
    DEFAULT_REPLY_TOKENS,
    OVERFLOW_TURN,
    STOP_REASONS,
    TRUNCATED_TURNS,
    TURNS_PLANNED,
    fits,
    slide_window,
)


def without_reasoning(messages: list) -> list:
    """``messages`` with the reasoning taken out of every assistant message.

    Inspect sends an earlier turn's reasoning back to the provider, and whether the model then
    sees it depends on its chat template: in the smoke test K2 Horizon's template kept it, while
    Granite's, Gemma 4's and Nemotron's dropped it (and Gemma 4's card says it must be dropped).
    Emptying it here gives every model the same history: its earlier answers only. The reasoning
    block is kept but empty, because some templates (K2 Horizon's) reject an assistant message
    with no thinking field at all; ``internal`` is kept so it still goes out as that field rather
    than as inline ``<think>`` text.
    """
    out = []
    for message in messages:
        if message.role == "assistant":
            parts = [ContentText(text=message.content)] if isinstance(message.content, str) else message.content
            content = [c.model_copy(update={"reasoning": "", "summary": None, "signature": None})
                       if c.type == "reasoning" else c for c in parts]
            if not any(c.type == "reasoning" for c in content):
                # A reply can come back with no reasoning at all; K2's template still needs the field
                # (the full run's cb-0175 failed on exactly that).
                content.insert(0, ContentReasoning(reasoning="", internal="reasoning"))
            message = message.model_copy(update={"content": content})
        out.append(message)
    return out


@solver
def pressure_turns(
    pressure_ids: list[str] | None = None,
    context_tokens: int | None = None,
    max_tokens: int | None = None,
    context_overflow: str = DEFAULT_CONTEXT_OVERFLOW,
):
    """Answer the question, then each of the item's ``pressure_turns`` in order.

    The model's own replies stay in context between turns, without their reasoning
    (``without_reasoning``); the transcript keeps the reasoning. Items without pressure turns
    get a single generation, so single-turn items run exactly as a plain question would.

    ``pressure_ids`` applies the repeated-sampling protocol in SCHEMA.md: every epoch
    samples the opening turn, but the pressure sequence plays only in epoch 1, except for
    the listed sample ids, which play it in every epoch. ``None`` plays it in every epoch
    for every item.

    ``context_tokens`` is the token budget for the conversation (see ``cb_budget``). A turn
    whose prompt plus ``max_tokens`` of reply would not fit is never sent whole, since a
    provider may silently drop the start of it. With ``context_overflow="slide"`` (default)
    the oldest exchanges are left out until it fits, so the window keeps the most recent
    turns, and the turn is recorded in ``context_clipped``. With ``"stop"``, or when even the
    latest message alone doesn't fit, the sequence stops there and the turn is recorded in
    the sample's metadata for the scorer. ``None`` turns the check off.

    Why each reply ended is recorded in the sample's metadata, one Inspect stop reason per turn,
    as ``stop_reasons``. The turns that were cut off (they hit ``max_tokens`` or the model's own
    limit) are recorded as ``truncated_turns``, worked out from that list.
    """
    if context_overflow not in CONTEXT_OVERFLOW_MODES:
        raise ValueError(f"context_overflow must be one of {CONTEXT_OVERFLOW_MODES}, got {context_overflow!r}")
    repeat = None if pressure_ids is None else set(pressure_ids)
    reply_tokens = max_tokens or DEFAULT_REPLY_TOKENS

    async def generate_if_it_fits(state: TaskState, generate: Generate) -> tuple[TaskState, bool]:
        turn = sum(1 for m in state.messages if m.role == "assistant")
        sent = without_reasoning(state.messages)
        window = sent
        if not await fits(get_model(), window, reply_tokens, context_tokens):
            window = None
            if context_overflow == "slide":
                window = await slide_window(get_model(), sent, reply_tokens, context_tokens)
            if window is None:
                state.metadata[OVERFLOW_TURN] = turn
                return state, False
            # Send only the most recent part of the conversation; the transcript keeps all of it.
            state.metadata.setdefault(CONTEXT_CLIPPED, []).append({"turn": turn, "dropped": len(sent) - len(window)})
        # The model is sent its earlier answers without their reasoning; the transcript keeps both.
        full = list(state.messages)
        state.messages = window
        state = await generate(state)
        state.messages = full + state.messages[len(window):]
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
