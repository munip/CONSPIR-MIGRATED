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

# Why a turn's label is "invalid" (Score.metadata["invalid_reasons"]).
MODEL_CONTEXT_OVERFLOW = "model_context_overflow"
JUDGE_CONTEXT_OVERFLOW = "judge_context_overflow"
UNPARSEABLE = "unparseable"


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
