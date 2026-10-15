#!/usr/bin/env python3
"""Tests for cut-off replies and the per-reason invalid breakdown. No API calls.

    python -m unittest discover -s inspect_eval -p "test_*.py"

A reply that hits ``max_tokens`` partway through has text, so it is not caught as an empty
reply, but it may stop before its conclusion. These check that the solver records it, that
the scorer does not grade it as if it were complete, and that ``report.py`` says how many
turns were invalid and why.
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))

from needs_tokenizer import needs_tokenizer
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

import conspiracybench as cb
import report

MULTITURN_CASES = [
    json.loads(line)
    for line in (HERE.parent / "eval" / "judge_cases_multiturn.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
WTC_ITEM = next(c["item"] for c in MULTITURN_CASES if "Building 7" in c["item"]["question"])
SINGLE_TURN = {k: v for k, v in WTC_ITEM.items() if k != "pressure_turns"}


def model_with(outputs):
    """Model under test: plays (text, stop_reason) pairs in order."""
    return get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", text, stop_reason=stop) for text, stop in outputs],
        memoize=False,
    )


def judge_saying(label, seen):
    def respond(input, tools, tool_choice, config):
        seen.append(input[-1].text)
        return ModelOutput.from_content("mockllm/model", label)

    return get_model("mockllm/model", custom_outputs=respond, memoize=False)


def run(item, outputs, seen, **task_args):
    """Run one item; return its score, sample metadata and report summary."""
    task = cb.build_task([cb.item_to_sample(item, "case")], **task_args)
    with tempfile.TemporaryDirectory() as tmp:
        [log] = eval(task, model=model_with(outputs), model_roles={"grader": judge_saying("hold", seen)},
                     display="none", log_dir=tmp, max_samples=1)
        sample = log.samples[0]
        score = sample.scores["conspiracy_judge"]
        json_path = report.report(Path(log.location), Path(tmp) / "results")
        summary = json.loads(json_path.read_text(encoding="utf-8"))
        csv_text = next((Path(tmp) / "results").glob("*.csv")).read_text(encoding="utf-8")
    return {"value": score.value, "score_metadata": score.metadata, "metadata": sample.metadata,
            "summary": summary, "csv": csv_text}


@needs_tokenizer
class TruncatedReplyTest(unittest.TestCase):
    def test_cut_off_reply_is_invalid_and_not_sent_to_the_judge(self):
        seen = []
        result = run(SINGLE_TURN, [("Uncontrolled fires started by debris from the North", "max_tokens")], seen)

        self.assertEqual(result["metadata"]["truncated_turns"], [0])
        self.assertEqual(result["value"], "invalid")
        self.assertEqual(result["score_metadata"]["invalid_reasons"], ["truncated_reply"])
        self.assertEqual(seen, [])  # an answer that stops mid-sentence is not graded as complete

    def test_only_the_cut_off_turn_is_invalid(self):
        seen = []
        result = run(WTC_ITEM, [("Fires.", "stop"), ("It is true that some engineers", "max_tokens"), ("Fires.", "stop")], seen)

        self.assertEqual(result["metadata"]["truncated_turns"], [1])
        self.assertEqual(result["value"], "hold>invalid>hold")
        self.assertEqual(result["score_metadata"]["invalid_reasons"], [None, "truncated_reply", None])
        self.assertEqual(len(seen), 2)

    def test_reply_stopped_by_the_models_own_limit_counts_as_cut_off(self):
        result = run(SINGLE_TURN, [("Uncontrolled fires started by", "model_length")], [])

        self.assertEqual(result["score_metadata"]["invalid_reasons"], ["truncated_reply"])

    def test_grade_truncated_sends_it_to_the_judge_but_still_counts_it(self):
        seen = []
        result = run(SINGLE_TURN, [("Uncontrolled fires started by debris from the North", "max_tokens")], seen,
                     grade_truncated=True)

        self.assertEqual(result["value"], "hold")
        self.assertEqual(len(seen), 1)
        self.assertEqual(result["score_metadata"]["truncated_turns"], [0])
        self.assertEqual(result["summary"]["invalid_turns"]["truncated_graded"]["count"], 1)
        self.assertEqual(result["summary"]["invalid_turns"]["invalid"]["count"], 0)

    def test_empty_reply_keeps_its_own_reason(self):
        """An empty reply is usually also cut off; it stays ``empty_reply`` as before."""
        result = run(SINGLE_TURN, [("", "max_tokens")], [])

        self.assertEqual(result["score_metadata"]["invalid_reasons"], ["empty_reply"])

    def test_a_reply_that_finished_is_not_flagged(self):
        seen = []
        result = run(SINGLE_TURN, [("Uncontrolled fires.", "stop")], seen)

        self.assertNotIn("truncated_turns", result["metadata"])
        self.assertEqual(result["value"], "hold")
        self.assertEqual(result["score_metadata"]["truncated_turns"], [])


@needs_tokenizer
class StopReasonsTest(unittest.TestCase):
    """The solver records why every reply ended; the cut-off turns are worked out from that list."""

    def test_every_turn_records_why_its_reply_ended(self):
        result = run(WTC_ITEM, [("Fires.", "stop"), ("It is true that some engineers", "max_tokens"), ("Fires.", "stop")], [])

        self.assertEqual(result["metadata"]["stop_reasons"], ["stop", "max_tokens", "stop"])

    def test_a_single_turn_item_records_one(self):
        result = run(SINGLE_TURN, [("Uncontrolled fires.", "stop")], [])

        self.assertEqual(result["metadata"]["stop_reasons"], ["stop"])

    def test_later_epochs_that_play_only_the_opening_turn_record_one(self):
        task = cb.build_task([cb.item_to_sample(WTC_ITEM, "case")], epochs=2, pressure_ids=[])
        outputs = [("Fires.", "stop"), ("Fires.", "stop"), ("Fires.", "stop"), ("Fires.", "stop")]
        with tempfile.TemporaryDirectory() as tmp:
            [log] = eval(task, model=model_with(outputs), model_roles={"grader": judge_saying("hold", [])},
                         display="none", log_dir=tmp, max_samples=1)
            recorded = {s.epoch: s.metadata["stop_reasons"] for s in log.samples}

        self.assertEqual(recorded, {1: ["stop", "stop", "stop"], 2: ["stop"]})

    def test_truncated_turns_are_the_cut_off_entries_of_the_list(self):
        cases = [
            (["stop", "stop", "stop"], None),
            (["max_tokens", "stop", "model_length"], [0, 2]),
            (["stop", "max_tokens", "max_tokens"], [1, 2]),
            (["content_filter", "stop", "unknown"], None),  # other stop reasons are not cut-offs
        ]
        for reasons, expected in cases:
            with self.subTest(reasons=reasons):
                result = run(WTC_ITEM, [("An answer.", reason) for reason in reasons], [], grade_truncated=True)

                self.assertEqual(result["metadata"]["stop_reasons"], reasons)
                if expected is None:
                    self.assertNotIn("truncated_turns", result["metadata"])
                else:
                    self.assertEqual(result["metadata"]["truncated_turns"], expected)


class InvalidTurnsReportTest(unittest.TestCase):
    @staticmethod
    def row(trajectory, reasons=None, truncated=None):
        row = {"id": "i", "epoch": 1, "trajectory": trajectory.split(">"), "cluster": "item:i",
               "category": "c", "band": "easy", "framing": None, "region": None, "multiturn": False}
        if reasons is not None:
            row["invalid_reasons"] = reasons
        if truncated is not None:
            row["truncated_turns"] = truncated
        return row

    def test_counts_turns_by_reason(self):
        rows = [
            self.row("hold>invalid>hold", [None, "truncated_reply", None]),
            self.row("invalid", ["empty_reply"]),
            self.row("hold>invalid", [None, "truncated_reply"]),
            self.row("hold"),
        ]
        turns = report.invalid_turns(rows)

        self.assertEqual(turns["turns"], 7)
        self.assertEqual(turns["invalid"]["count"], 3)
        self.assertEqual({r: m["count"] for r, m in turns["by_reason"].items()},
                         {"empty_reply": 1, "truncated_reply": 2})
        self.assertAlmostEqual(turns["by_reason"]["truncated_reply"]["rate"], 2 / 7)
        self.assertEqual(sum(m["count"] for m in turns["by_reason"].values()), turns["invalid"]["count"])

    def test_log_without_recorded_reasons_is_counted_as_unrecorded(self):
        """Logs made before reasons existed still get an honest count, not a crash."""
        turns = report.invalid_turns([self.row("hold>invalid"), self.row("invalid", [])])

        self.assertEqual({r: m["count"] for r, m in turns["by_reason"].items()}, {"unrecorded": 2})

    def test_no_turns_gives_no_rate(self):
        turns = report.invalid_turns([])

        self.assertEqual(turns["turns"], 0)
        self.assertIsNone(turns["invalid"]["rate"])
        self.assertEqual(turns["by_reason"], {})

    @needs_tokenizer
    def test_report_files_and_console_note_carry_the_breakdown(self):
        seen = []
        result = run(WTC_ITEM, [("Fires.", "stop"), ("It is true that some engineers", "max_tokens"), ("Fires.", "stop")], seen)
        turns = result["summary"]["invalid_turns"]

        self.assertEqual(turns["turns"], 3)
        self.assertEqual(turns["by_reason"]["truncated_reply"]["count"], 1)
        self.assertIn("all,invalid_turn_rate,", result["csv"])
        self.assertIn("all,invalid_turn_rate[truncated_reply],", result["csv"])

    @needs_tokenizer
    def test_main_prints_a_note_naming_the_reasons(self):
        task = cb.build_task([cb.item_to_sample(SINGLE_TURN, "case")])
        with tempfile.TemporaryDirectory() as tmp:
            [log] = eval(task, model=model_with([("Uncontrolled fires started by", "max_tokens")]),
                         model_roles={"grader": judge_saying("hold", [])}, display="none", log_dir=tmp, max_samples=1)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                report.main([log.location, "--out", str(Path(tmp) / "results")])

        self.assertIn("1 of 1 turns invalid (1 truncated_reply)", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
