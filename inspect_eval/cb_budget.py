"""A token budget for every prompt, checked before the prompt is sent.

Some providers cut an over-long prompt instead of failing. Ollama does, through both its
native and OpenAI-compatible endpoints, and it drops the *start* of the prompt (see #68).
For the judge, the start holds the instructions, the question, the ground truth and the
``diverges_if`` checklist, so a truncated judge prompt still returns a confident label,
chosen without the ground truth. For the model under test, the start of a long pressure
sequence is the original question.

Whether a prompt overflows depends on the machine as well as the model: Ollama picks its
default context length from available memory, so one run can error on one laptop and
truncate silently on another. The budget is therefore a value the task sets, not
something read from the provider.

The check is made *before* sending, from Inspect's own estimate (``Model.count_tokens``).
Checking afterwards against the provider's reported input tokens does not work: Ollama
reports only tokens that were not already in its prompt cache, so the second and third
turns of a conversation report about one token each.
"""
from __future__ import annotations

from inspect_ai.model import Model

# Matches num_ctx in inspect_eval/ollama/*.Modelfile. A local run must load the model
# with at least this context, or the budget promises room the model does not have.
DEFAULT_CONTEXT_TOKENS = 16384

# Room kept for the reply when the task sets no max_tokens.
DEFAULT_REPLY_TOKENS = 2048

# Room kept for the judge's reply: a label, plus the justification small judges add.
JUDGE_REPLY_TOKENS = 256

# Sample metadata key: the turn the model under test was not asked, for lack of room.
OVERFLOW_TURN = "context_overflow_turn"

# Sample metadata key: how many turns the solver meant to play for this sample (the opening
# turn, plus the pressure turns when the sequence plays). Lets the scorer tell a transcript
# that stopped early from one that was meant to be short.
TURNS_PLANNED = "turns_planned"

# Per-turn labels are joined into one scalar score value (see scorer/cb_scorer.py), and a turn that
# can't be graded is labelled INVALID.
TURN_SEPARATOR = ">"
INVALID = "invalid"

# Why a turn's label is "invalid" (Score.metadata["invalid_reasons"]).
MODEL_CONTEXT_OVERFLOW = "model_context_overflow"
JUDGE_CONTEXT_OVERFLOW = "judge_context_overflow"
UNPARSEABLE = "unparseable"
INCOMPLETE_TRANSCRIPT = "incomplete_transcript"
# The model under test gave no reply text, e.g. a thinking model whose reasoning used the
# whole max_tokens. There is nothing to grade, and judges label it confidently anyway.
EMPTY_REPLY = "empty_reply"
# The reply has text but was cut off before the model finished it.
TRUNCATED_REPLY = "truncated_reply"

# Sample metadata key: why each model reply ended, one Inspect stop reason per turn, as Inspect
# reports it (`stop`, `max_tokens`, `model_length`, `tool_calls`, `content_filter`, `unknown`).
STOP_REASONS = "stop_reasons"

# Sample metadata key: the turns whose reply was cut off, recorded by the solver. It is worked out
# from STOP_REASONS each time a reply arrives, so the two cannot disagree.
TRUNCATED_TURNS = "truncated_turns"

# Inspect stop reasons that mean the reply ended because it ran out of room, not because the
# model finished: the task's max_tokens, or the model's own context limit.
CUT_OFF_STOP_REASONS = ("max_tokens", "model_length")


async def fits(model: Model, prompt, reply_tokens: int, context_tokens: int | None) -> bool:
    """Whether ``prompt`` plus room for a reply fits in ``context_tokens``.

    ``prompt`` is what would be sent: a string or a list of chat messages.
    ``context_tokens=None`` turns the check off. The estimate comes from Inspect, which
    uses the provider's tokenizer when it has one and a generic tokenizer with a safety
    margin otherwise, so it is deliberately a little conservative.
    """
    if context_tokens is None:
        return True
    return await model.count_tokens(prompt) + reply_tokens <= context_tokens


# What happens to a conversation that no longer fits: "slide" drops the oldest exchanges until it
# does (the default), "stop" ends the sequence there and scores the turn invalid.
CONTEXT_OVERFLOW_MODES = ("slide", "stop")
DEFAULT_CONTEXT_OVERFLOW = "slide"

# Sample metadata keys: the turns sent with a clipped conversation, as {"turn", "dropped"} records
# (dropped = earlier messages left out), for the model under test and for the judge.
CONTEXT_CLIPPED = "context_clipped"
JUDGE_CONTEXT_CLIPPED = "judge_context_clipped"


async def slide_window(model: Model, messages: list, reply_tokens: int, context_tokens: int | None):
    """The most recent part of ``messages`` that fits, or None if even the latest message alone doesn't.

    Leading system messages always stay. After them, the oldest messages are dropped two at a time
    (one user message and the reply to it), so the window still starts with a user message and
    always ends with the latest one. The full conversation stays in the transcript; only what is
    sent is clipped.
    """
    head = 0
    while head < len(messages) and _role(messages[head]) == "system":
        head += 1
    system, rest = list(messages[:head]), list(messages[head:])
    while rest:
        window = system + rest
        if await fits(model, window, reply_tokens, context_tokens):
            return window
        if len(rest) <= 1:
            return None
        rest = rest[2:]
    return None


def _role(message):
    return message["role"] if isinstance(message, dict) else message.role
