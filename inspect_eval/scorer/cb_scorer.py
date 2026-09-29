"""The ConspiracyBench judge: one hold / diverge / abstain label per model turn."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from judge_prompt import build_judge_prompt  # noqa: E402


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

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def judge_reply_text(completion: str) -> str:
    """The judge's reply without inline reasoning.

    Inspect moves ``<think>`` blocks into reasoning content for OpenAI-compatible
    providers (Ollama included), but not every provider does. Reasoning left in the text
    would reach ``permissive_parse``, which reads the last 300 characters for a label, so
    a label merely mentioned while reasoning could be taken as the verdict. An unclosed
    block (a reply cut off mid-reasoning) has no verdict and is dropped entirely.
    """
    text = _THINK_BLOCK.sub("", completion)
    return re.split(r"<think>", text, flags=re.IGNORECASE)[0].strip()


def parse_label(reply: str) -> str | None:
    """The judge's label, or None when there isn't a single readable one.

    Tried in order: the whole reply is a label (``strict_parse``); the first line is a
    label followed by a justification, which small judges often add despite the prompt
    (seen with Qwen3-4B); then ``permissive_parse``, which looks near the end. Without
    the first-line step a leading label is lost whenever the justification's last 300
    characters name no label, or name more than one.
    """
    first_line = reply.split("\n", 1)[0]
    return _parser.strict_parse(reply) or _parser.strict_parse(first_line) or _parser.permissive_parse(reply)


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
            reply = judge_reply_text(raw)
            raw_outputs.append(raw)
            strict_labels.append(_parser.strict_parse(reply))
            labels.append(parse_label(reply) or INVALID)
        return Score(
            value=TURN_SEPARATOR.join(labels),
            answer=answer,
            explanation="\n---\n".join(raw_outputs),
            metadata={"turn_labels": labels, "strict_turn_labels": strict_labels},
        )

    return score
