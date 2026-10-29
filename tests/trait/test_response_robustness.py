import json
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

from inspect_eval.trait.response_robustness import parse_judgment, _score_response_robustness


def test_parse_judgment_accepts_three_independent_dimensions():
    result = parse_judgment(
        '```json\n{"factual_alignment":"contradicts",'
        '"conspiracy_endorsement":"does_not_endorse","abstention":false}\n```'
    )
    assert result == {
        "factual_alignment": "contradicts",
        "conspiracy_endorsement": "does_not_endorse",
        "abstention": False,
    }


@pytest.mark.parametrize("payload", [
    {"factual_alignment": "unknown", "conspiracy_endorsement": "ambiguous", "abstention": False},
    {"factual_alignment": "matches", "conspiracy_endorsement": "endorses", "abstention": "false"},
    {"factual_alignment": "matches", "conspiracy_endorsement": "endorses", "abstention": False, "extra": 1},
])
def test_parse_judgment_rejects_invalid_labels_and_shapes(payload):
    with pytest.raises(ValueError):
        parse_judgment(json.dumps(payload))


def test_response_scorer_uses_a_grader_and_records_all_dimensions():
    prompts = []

    class FakeJudge:
        async def generate(self, prompt, config):
            prompts.append(prompt)
            return SimpleNamespace(completion=json.dumps({
                "factual_alignment": "matches",
                "conspiracy_endorsement": "ambiguous",
                "abstention": False,
            }))

    state = SimpleNamespace(
        metadata={"item": {"question": "Question?", "ground_truth_answer": "Ground truth."}},
        messages=[ChatMessageUser(content="Question?"), ChatMessageAssistant(content="Answer.")],
    )
    with patch("inspect_eval.trait.response_robustness.get_model", return_value=FakeJudge()) as get_model:
        score = asyncio.run(_score_response_robustness(state, "grader"))

    get_model.assert_called_once_with(role="grader", required=True)
    assert "Ground truth." in prompts[0]
    assert "Answer." in prompts[0]
    assert json.loads(score.value) == [{
        "factual_alignment": "matches",
        "conspiracy_endorsement": "ambiguous",
        "abstention": False,
    }]
