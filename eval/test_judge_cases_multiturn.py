#!/usr/bin/env python3
"""Structural checks for eval/judge_cases_multiturn.jsonl. Run from the repo root:

    python -m unittest discover -s eval -p "test_*.py"

These do not call a judge. They make sure every case is well formed, so that a
judge run over the file is grading what each case says it is grading.
"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from judge_prompt import build_judge_prompt


CASES_PATH = HERE / "judge_cases_multiturn.jsonl"
VALID_LABELS = {"hold", "diverge", "abstain"}
REQUIRED = {"case_id", "item", "transcript", "model_answer", "expected_label", "purpose"}


def load_cases():
    cases = []
    for lineno, line in enumerate(CASES_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            cases.append((lineno, json.loads(line)))
    return cases


class MultiTurnJudgeCasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()

    def test_file_is_not_empty(self):
        self.assertGreater(len(self.cases), 0)

    def test_required_fields_and_labels(self):
        for lineno, case in self.cases:
            with self.subTest(line=lineno):
                self.assertEqual(REQUIRED - set(case), set(), "missing fields")
                self.assertIn(case["expected_label"], VALID_LABELS)
                self.assertTrue(case["model_answer"].strip())
                self.assertTrue(case["purpose"].strip())
                for field in ("question", "ground_truth_answer", "pressure_turns"):
                    self.assertIn(field, case["item"])

    def test_case_ids_are_unique(self):
        ids = [case["case_id"] for _, case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_label_is_covered(self):
        """A set with no abstain cases cannot catch a judge that never abstains."""
        labels = {case["expected_label"] for _, case in self.cases}
        self.assertEqual(labels, VALID_LABELS)

    def test_transcript_shape(self):
        """Question first, alternating roles, ending on the user turn being answered."""
        for lineno, case in self.cases:
            transcript = case["transcript"]
            with self.subTest(line=lineno, case=case["case_id"]):
                self.assertGreaterEqual(len(transcript), 3, "a multi-turn case needs at least one pressure turn")
                self.assertEqual(transcript[0], {"role": "user", "content": case["item"]["question"]})
                for i, message in enumerate(transcript):
                    self.assertEqual(set(message), {"role", "content"})
                    self.assertEqual(message["role"], "user" if i % 2 == 0 else "assistant")
                    self.assertTrue(message["content"].strip())
                self.assertEqual(transcript[-1]["role"], "user")

    def test_user_turns_follow_the_items_pressure_turns(self):
        """The scripted turns in a case must be the item's own, in order.

        Otherwise a case could quietly test pushback the item never asks for.
        """
        for lineno, case in self.cases:
            with self.subTest(line=lineno, case=case["case_id"]):
                user_turns = [m["content"] for m in case["transcript"][2::2]]
                pressure = case["item"]["pressure_turns"]
                self.assertLessEqual(len(user_turns), len(pressure))
                self.assertEqual(user_turns, pressure[: len(user_turns)])

    def test_prompt_renders_conversation_and_final_answer(self):
        for lineno, case in self.cases:
            with self.subTest(line=lineno, case=case["case_id"]):
                prompt = build_judge_prompt(case["item"], case["model_answer"], case["transcript"])
                self.assertIn("Conversation before the response you are grading.", prompt)
                self.assertIn(f"Model response:\n{case['model_answer']}", prompt)
                for message in case["transcript"]:
                    self.assertIn(message["content"], prompt)


if __name__ == "__main__":
    unittest.main()
