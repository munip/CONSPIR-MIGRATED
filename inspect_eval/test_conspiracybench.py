#!/usr/bin/env python3
"""Tests for inspect_eval/conspiracybench.py. Needs inspect-ai; makes no API calls.

    python -m unittest discover -s inspect_eval -p "test_*.py"

Both the model under test and the judge are Inspect's mockllm, scripted per test, so these
check the plumbing (what the judge is shown, how labels become trajectories and rates),
not any real model's behaviour.
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "eval"))

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

import conspiracybench as cb
from judge_prompt import build_judge_prompt

MULTITURN_CASES = [
    json.loads(line)
    for line in (ROOT / "eval" / "judge_cases_multiturn.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
MMR_ITEM = next(c["item"] for c in MULTITURN_CASES if "MMR" in c["item"]["question"])
WTC_ITEM = next(c["item"] for c in MULTITURN_CASES if "Building 7" in c["item"]["question"])


def scripted_model(answers):
    """Model under test: returns the given answers in order."""
    return get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", a) for a in answers],
        memoize=False,
    )


def recording_judge(label_for_prompt, seen):
    """Judge: records every prompt it receives and answers with label_for_prompt(prompt)."""

    def respond(input, tools, tool_choice, config):
        prompt = input[-1].text
        seen.append(prompt)
        return ModelOutput.from_content("mockllm/model", label_for_prompt(prompt))

    return get_model("mockllm/model", custom_outputs=respond, memoize=False)


def run(item, answers, judge, epochs=1):
    """Run one item and return what the tests inspect, read while the log still exists."""
    task = cb.build_task([cb.item_to_sample(item, "case")], epochs=epochs)
    with tempfile.TemporaryDirectory() as log_dir:
        [log] = eval(
            task,
            model=scripted_model(answers),
            model_roles={"grader": judge},
            display="none",
            log_dir=log_dir,
            max_samples=1,
        )
        return {
            "status": log.status,
            # One entry per sample per epoch, before reduction.
            "values": [sample.scores["conspiracy_judge"].value for sample in log.samples],
            # After the `collect` reducer: one list of per-epoch values per sample.
            "reduced": [r.value for red in (log.reductions or []) for r in red.samples],
            "metrics": {name: m.value for s in log.results.scores for name, m in s.metrics.items()},
        }


class DatasetTest(unittest.TestCase):
    def test_loads_corpus_and_namespaces_pending_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "questions" / "pending").mkdir(parents=True)
            corpus_item = {"id": "cb-0001", "question": "Q1", "ground_truth_answer": "A1"}
            pending_items = [
                {"id": "cb-pending", "question": "Q2", "ground_truth_answer": "A2"},
                {"id": "cb-pending", "question": "Q3", "ground_truth_answer": "A3"},
            ]
            (root / "questions" / "corpus.jsonl").write_text(json.dumps(corpus_item) + "\n", encoding="utf-8")
            (root / "questions" / "pending" / "batch.jsonl").write_text(
                "\n".join(json.dumps(i) for i in pending_items) + "\n", encoding="utf-8"
            )
            (root / "questions" / "pending" / "README.md").write_text("not data", encoding="utf-8")

            samples = list(cb.conspiracybench_dataset(root))
            self.assertEqual([s.id for s in samples], ["cb-0001", "pending/batch:1", "pending/batch:2"])
            self.assertEqual(samples[1].input, "Q2")
            self.assertEqual(samples[1].target, "A2")
            self.assertEqual(samples[1].metadata["item"], pending_items[0])

            corpus_only = list(cb.conspiracybench_dataset(root, include_pending=False))
            self.assertEqual([s.id for s in corpus_only], ["cb-0001"])

    def test_real_repo_loads(self):
        samples = list(cb.conspiracybench_dataset())
        self.assertGreater(len(samples), 0)
        self.assertEqual(len({s.id for s in samples}), len(samples))


class JudgePromptParityTest(unittest.TestCase):
    """The judge must see exactly what eval/judge_prompt.py builds, turn by turn."""

    def test_multiturn_prompts_match_existing_builder(self):
        answers = ["No link.", "Still no link.", "The large studies covered risk groups.", "You raise a fair point."]
        labels = iter(["hold", "hold", "hold", "diverge"])
        seen = []
        log = run(MMR_ITEM, answers, recording_judge(lambda _: next(labels), seen))

        q, p = MMR_ITEM["question"], MMR_ITEM["pressure_turns"]
        expected = [build_judge_prompt(MMR_ITEM, answers[0], None)]
        history = [{"role": "user", "content": q}]
        for i in range(3):
            history += [{"role": "assistant", "content": answers[i]}, {"role": "user", "content": p[i]}]
            expected.append(build_judge_prompt(MMR_ITEM, answers[i + 1], list(history)))

        self.assertEqual(seen, expected)
        self.assertEqual(log["values"], ["hold>hold>hold>diverge"])

    def test_single_turn_item_uses_the_single_turn_prompt(self):
        item = {k: v for k, v in WTC_ITEM.items() if k != "pressure_turns"}
        seen = []
        log = run(item, ["Uncontrolled fires."], recording_judge(lambda _: "hold", seen))

        self.assertEqual(seen, [build_judge_prompt(item, "Uncontrolled fires.")])
        self.assertNotIn("Conversation before", seen[0])
        self.assertEqual(log["values"], ["hold"])


class ScoringTest(unittest.TestCase):
    def test_concession_is_measured_on_multiturn_trajectories(self):
        log = run(WTC_ITEM, ["Fires.", "Fires, still.", "It is disputed."],
                  recording_judge(lambda p: "diverge" if "It is disputed." in p else "hold", []))
        m = log["metrics"]
        self.assertEqual(m["divergence_rate"], 0.0)  # opening turn held
        self.assertEqual(m["concession_rate"], 1.0)
        self.assertEqual(m["retreat_rate"], 0.0)
        self.assertEqual(m["invalid_rate"], 0.0)

    def test_every_epoch_keeps_its_own_trajectory(self):
        """Two epochs with different outcomes must both survive reduction.

        Score.metadata is reduced to the first epoch, and a dict value would be reduced
        key by key; either would lose or mix the second trajectory.
        """
        one_turn_item = {**WTC_ITEM, "pressure_turns": WTC_ITEM["pressure_turns"][:1]}
        answers = ["Fires.", "Fires, still.", "Fires.", "It is disputed."]
        log = run(one_turn_item, answers,
                  recording_judge(lambda p: "diverge" if "It is disputed." in p else "hold", []),
                  epochs=2)
        self.assertEqual(sorted(log["values"]), ["hold>diverge", "hold>hold"])
        [reduced] = log["reduced"]
        self.assertEqual(sorted(reduced), ["hold>diverge", "hold>hold"])
        self.assertEqual(log["metrics"]["concession_rate"], 0.5)

    def test_unparseable_judge_output_is_reported_not_guessed(self):
        log = run({k: v for k, v in WTC_ITEM.items() if k != "pressure_turns"}, ["Fires."],
                  recording_judge(lambda _: "I am not sure how to label this.", []))
        m = log["metrics"]
        self.assertEqual(log["values"], ["invalid"])
        self.assertEqual(m["invalid_rate"], 1.0)
        self.assertTrue(math.isnan(m["divergence_rate"]))

    def test_judge_is_a_required_role(self):
        """Without a grader role the run must fail, not grade the model with itself.

        The model under test is given enough outputs to answer all three turns and then
        act as its own judge, so the only thing that can fail the run is the missing role.
        """
        task = cb.build_task([cb.item_to_sample(WTC_ITEM, "case")])
        with tempfile.TemporaryDirectory() as log_dir:
            [log] = eval(task, model=scripted_model(["a", "b", "c", "hold", "hold", "hold"]),
                         display="none", log_dir=log_dir)
            status, error = log.status, str(log.error or "")
        self.assertEqual(status, "error")
        self.assertIn("grader", error)


if __name__ == "__main__":
    unittest.main()
