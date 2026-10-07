"""Tests for speedtest.py's prompt check. No model calls; request bodies are shaped like
the ones Inspect logs for Ollama's OpenAI-compatible endpoint."""
import unittest
from types import SimpleNamespace

from speedtest import chat_call, request_key

REPLY = {"role": "assistant", "content": "Oswald.", "reasoning": "Thinking..."}


def request(question="Who shot JFK?", turn2="Are you sure?", max_tokens=2048, reply=None, rid="a"):
    sent = reply or {"role": "assistant", "content": "\nOswald.", "reasoning": "Thinking..."}
    return {
        "model": "cb-qwen3.5-4b",
        "max_tokens": max_tokens,
        "extra_headers": {"x-irid": rid},
        "messages": [{"role": "user", "content": question}, sent, {"role": "user", "content": turn2}],
    }


class RequestKeyTest(unittest.TestCase):
    def test_same_request_with_a_different_reply_and_request_id_matches(self):
        other_reply = {"role": "assistant", "content": "Lee Harvey Oswald.", "reasoning": "Hmm."}
        other_sent = {"role": "assistant", "content": "\nLee Harvey Oswald.", "reasoning": "Hmm."}
        a, problems_a = request_key(request(), [REPLY])
        b, problems_b = request_key(request(reply=other_sent, rid="b"), [other_reply])
        self.assertEqual((problems_a, problems_b), ([], []))
        self.assertEqual(a, b)

    def test_changed_prompt_or_setting_differs(self):
        base, _ = request_key(request(), [REPLY])
        for changed in (request(question="Who killed JFK?"), request(turn2="Really?"), request(max_tokens=4096)):
            self.assertNotEqual(request_key(changed, [REPLY])[0], base)

    def test_reply_in_context_must_be_the_models_own(self):
        tampered = {"role": "assistant", "content": "\nSomeone else.", "reasoning": "Thinking..."}
        _, problems = request_key(request(reply=tampered), [REPLY])
        self.assertEqual(problems, ["a reply in context is not the model's own"])

    def test_dropped_reply_is_reported(self):
        _, problems = request_key(request(), [])
        self.assertEqual(problems, ["1 replies in context, expected 0"])


class ChatCallTest(unittest.TestCase):
    def test_openai_style_call_gives_request_and_reply(self):
        event = SimpleNamespace(call=SimpleNamespace(request=request(), response={"choices": [{"message": REPLY}]}))
        self.assertEqual(chat_call(event), (request(), REPLY))

    def test_other_providers_give_a_clear_error(self):
        # mockllm logs no call; other providers log bodies without messages / choices.
        for call in (None, SimpleNamespace(request={"input": []}, response={"output": []})):
            with self.assertRaisesRegex(ValueError, "OpenAI-compatible"):
                chat_call(SimpleNamespace(call=call))


if __name__ == "__main__":
    unittest.main()
