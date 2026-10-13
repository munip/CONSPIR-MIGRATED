"""Tests for speedtest.py's prompt check. No model calls; request bodies are shaped like
the ones Inspect logs for Ollama's OpenAI-compatible endpoint."""
import tempfile
import unittest
from types import SimpleNamespace

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from speedtest import chat_call, compare_judged, report, request_key

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


def mock_log(log_samples=True):
    """A real log from a two-sample mockllm run, with or without stored samples."""
    task = Task(dataset=[Sample(input="a", id="a"), Sample(input="b", id="b")], solver=generate())
    with tempfile.TemporaryDirectory() as d:
        [log] = eval(task, model="mockllm/model", log_dir=d, log_samples=log_samples, display="none")
        return report(log.location)


class ReportTest(unittest.TestCase):
    def test_run_with_samples_has_no_warning_and_lists_slowest(self):
        r = mock_log()
        self.assertEqual(r["warnings"], [])
        self.assertEqual(r["samples"], 2)
        self.assertEqual(sorted(s["id"] for s in r["slowest_samples"]), ["a", "b"])

    def test_log_without_samples_warns_and_reports_null_not_zero(self):
        r = mock_log(log_samples=False)
        self.assertEqual(len(r["warnings"]), 1)
        self.assertIn("no samples", r["warnings"][0])
        # Exact from Inspect's totals, even with no stored samples.
        self.assertEqual((r["samples"], r["errors"]), (2, 0))
        # What needs the samples is unknown, not zero.
        for key in ("run_out_tok_per_s", "model_under_test", "by_group", "judge", "slowest_samples"):
            self.assertIsNone(r[key], key)


class CompareJudgedTest(unittest.TestCase):
    BASE = {("cb-0001", 1): ("hold>diverge", ["p1", "p2"]), ("cb-0002", 1): ("hold", ["p3"])}

    def test_identical_scoring_agrees(self):
        self.assertEqual(compare_judged(self.BASE, dict(self.BASE)), (3, 3, []))

    def test_changed_label_is_reported_and_counted(self):
        after = {**self.BASE, ("cb-0001", 1): ("hold>hold", ["p1", "p2"])}
        turns, agree, problems = compare_judged(self.BASE, after)
        self.assertEqual((turns, agree), (3, 2))
        self.assertEqual(problems, ["('cb-0001', 1): labels hold>diverge -> hold>hold"])

    def test_changed_judge_prompt_is_reported_even_with_the_same_label(self):
        after = {**self.BASE, ("cb-0002", 1): ("hold", ["p3 edited"])}
        self.assertEqual(compare_judged(self.BASE, after)[2], ["('cb-0002', 1): judge prompts differ"])

    def test_missing_sample_is_reported(self):
        after = {("cb-0001", 1): self.BASE[("cb-0001", 1)]}
        self.assertEqual(compare_judged(self.BASE, after)[2], ["('cb-0002', 1): only in before"])


if __name__ == "__main__":
    unittest.main()
