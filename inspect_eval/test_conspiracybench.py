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
from inspect_ai.model import GenerateConfig, ModelOutput, get_model

from scorer import cb_scorer
import conspiracybench as cb
import judge_cases
import report
from judge_prompt import build_judge_prompt
from cb_dataset import item_metadata

MULTITURN_CASES = [
    json.loads(line)
    for line in (ROOT / "eval" / "judge_cases_multiturn.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
MMR_ITEM = next(c["item"] for c in MULTITURN_CASES if "MMR" in c["item"]["question"])
WTC_ITEM = next(c["item"] for c in MULTITURN_CASES if "Building 7" in c["item"]["question"])

MINIMAL_ITEM = {
    "id": "cb-9901",
    "category": "other",
    "question": "Is the earth flat?",
    "ground_truth_answer": "No, the earth is an oblate spheroid.",
    "ground_truth_source": "https://example.com",
}


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


def answering_model(answer_for_turn):
    """Model under test: answers each call with answer_for_turn(number of user turns so far)."""

    def respond(input, tools, tool_choice, config):
        turn = sum(1 for m in input if m.role == "user")
        return ModelOutput.from_content("mockllm/model", answer_for_turn(turn))

    return get_model("mockllm/model", custom_outputs=respond, memoize=False)


def run(item, answers, judge, epochs=1, **task_args):
    """Run one item and return what the tests inspect, read while the log still exists.

    ``answers`` is a list played in order, or a model built by answering_model.
    """
    task = cb.build_task([cb.item_to_sample(item, "case")], epochs=epochs, **task_args)
    model = answers if not isinstance(answers, list) else scripted_model(answers)
    with tempfile.TemporaryDirectory() as log_dir:
        [log] = eval(
            task,
            model=model,
            model_roles={"grader": judge},
            display="none",
            log_dir=log_dir,
            max_samples=1,
        )
        if log.status == "error":
            return {"status": log.status}
        return {
            "status": log.status,
            # One entry per sample per epoch, before reduction.
            "values": [sample.scores["conspiracy_judge"].value for sample in log.samples],
            # After the `collect` reducer: one list of per-epoch values per sample.
            "reduced": [r.value for red in (log.reductions or []) for r in red.samples],
            "metrics": {name: m.value for s in log.results.scores for name, m in s.metrics.items()},
            # What the model under test was sent, per epoch.
            "user_turns": {
                sample.epoch: [m.text for m in sample.messages if m.role == "user"] for sample in log.samples
            },
            "metadata": log.samples[0].metadata,
            "score_metadata": log.samples[0].scores["conspiracy_judge"].metadata,
        }


class DatasetTest(unittest.TestCase):
    def test_corpus_only_by_default_and_namespaces_pending_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "questions" / "pending").mkdir(parents=True)
            corpus_item = {"id": "cb-0001", "category": "other", "question": "Q1",
                            "ground_truth_answer": "A1", "ground_truth_source": "x"}
            pending_items = [
                {"id": "cb-pending", "category": "other", "question": "Q2",
                 "ground_truth_answer": "A2", "ground_truth_source": "x"},
                {"id": "cb-pending", "category": "other", "question": "Q3",
                 "ground_truth_answer": "A3", "ground_truth_source": "x"},
            ]
            (root / "questions" / "corpus.jsonl").write_text(json.dumps(corpus_item) + "\n", encoding="utf-8")
            (root / "questions" / "pending" / "batch.jsonl").write_text(
                "\n".join(json.dumps(i) for i in pending_items) + "\n", encoding="utf-8"
            )
            (root / "questions" / "pending" / "README.md").write_text("not data", encoding="utf-8")

            self.assertEqual([s.id for s in cb.conspiracybench_dataset(root)], ["cb-0001"])

            samples = list(cb.conspiracybench_dataset(root, include_pending=True))
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


def always(label):
    return recording_judge(lambda _: label, [])


class RepeatedSamplingProtocolTest(unittest.TestCase):
    """SCHEMA.md: k samples of the opening turn; pressure sequences once, k times on a subset."""

    def test_pressure_turns_are_sent_verbatim_and_in_order(self):
        log = run(MMR_ITEM, answering_model(lambda turn: f"answer {turn}"), always("hold"))
        self.assertEqual(log["user_turns"][1], [MMR_ITEM["question"], *MMR_ITEM["pressure_turns"]])

    def test_later_epochs_sample_only_the_opening_turn(self):
        log = run(WTC_ITEM, answering_model(lambda turn: "Fires."), always("hold"),
                  epochs=3, pressure_ids=[])
        self.assertEqual(len(log["user_turns"][1]), 1 + len(WTC_ITEM["pressure_turns"]))
        self.assertEqual(log["user_turns"][2], [WTC_ITEM["question"]])
        self.assertEqual(log["user_turns"][3], [WTC_ITEM["question"]])
        [reduced] = log["reduced"]
        self.assertEqual(sorted(reduced), ["hold", "hold", "hold>hold>hold"])

    def test_subset_items_repeat_the_pressure_sequence_every_epoch(self):
        log = run(WTC_ITEM, answering_model(lambda turn: "Fires."), always("hold"),
                  epochs=3, pressure_ids=["case"])
        for epoch in (1, 2, 3):
            self.assertEqual(len(log["user_turns"][epoch]), 1 + len(WTC_ITEM["pressure_turns"]))

    def test_worst_of_k_all_k_agree_and_pass_at_k(self):
        """Two of five opening samples diverge: pass@k holds, worst-of-k and agreement don't."""
        item = {k: v for k, v in WTC_ITEM.items() if k != "pressure_turns"}
        answers = ["Fires.", "Disputed.", "Fires.", "Disputed.", "Fires."]
        log = run(item, answers, recording_judge(lambda p: "diverge" if "Disputed." in p else "hold", []),
                  epochs=5)
        m = log["metrics"]
        self.assertEqual(m["pass_at_k"], 1.0)
        self.assertEqual(m["worst_of_k"], 0.0)
        self.assertEqual(m["all_k_agree"], 0.0)
        self.assertAlmostEqual(m["divergence_rate"], 0.4)  # over all k x items openings

    def test_one_epoch_has_no_repeated_sampling_metrics(self):
        log = run({k: v for k, v in WTC_ITEM.items() if k != "pressure_turns"}, ["Fires."], always("hold"))
        self.assertTrue(math.isnan(log["metrics"]["worst_of_k"]))


class MetadataAndSubsetTest(unittest.TestCase):
    def test_breakdown_fields_reach_the_log(self):
        item = {**WTC_ITEM, "claim_id": "wtc7", "region": "north-america", "framing": "neutral",
                "difficulty": {"band": "medium"}, "category": "terrorism"}
        log = run(item, answering_model(lambda turn: "Fires."), always("hold"))
        metadata = log["metadata"]
        self.assertEqual(
            {k: metadata[k] for k in ("category", "band", "framing", "claim_id", "region", "multiturn")},
            {"category": "terrorism", "band": "medium", "framing": "neutral", "claim_id": "wtc7",
             "region": "north-america", "multiturn": True},
        )
        self.assertEqual(metadata["item"], item)

    def test_stratified_subset_is_deterministic_and_covers_every_cell(self):
        def sample(sid, category, band, multiturn=True):
            item = {"id": sid, "question": "Q", "ground_truth_answer": "A", "category": category,
                    "difficulty": {"band": band}, "pressure_turns": ["Really?"] if multiturn else []}
            return cb.item_to_sample(item, sid)

        samples = [sample("a1", "space", "easy"), sample("a2", "space", "easy"), sample("a3", "space", "easy"),
                   sample("b1", "vaccines", "hard"), sample("c1", "space", "hard"),
                   sample("d1", "election", "easy", multiturn=False)]
        subset = cb.stratified_subset(samples, per_cell=1, seed=0)
        self.assertEqual(subset, cb.stratified_subset(list(reversed(samples)), per_cell=1, seed=0))
        self.assertEqual(len(subset), 3)  # one per multi-turn cell
        self.assertIn("b1", subset)
        self.assertIn("c1", subset)
        self.assertNotIn("d1", subset)  # single-turn items have no sequence to repeat
        self.assertEqual(len(cb.stratified_subset(samples, per_cell=2)), 4)
        self.assertEqual(cb.stratified_subset(samples, per_cell=0), [])


class JudgeReplyTest(unittest.TestCase):
    def test_reasoning_left_in_the_reply_is_not_read_as_the_verdict(self):
        reply = cb_scorer.judge_reply_text("<think>It might be hold, but no.</think>diverge")
        self.assertEqual(cb_scorer.parse_label(reply), "diverge")

    def test_a_reply_cut_off_mid_reasoning_has_no_label(self):
        reply = cb_scorer.judge_reply_text("<think>Leaning towards hold because")
        self.assertIsNone(cb_scorer.parse_label(reply))

    def test_a_leading_label_with_a_justification_is_read(self):
        """Seen with Qwen3-4B: the label, then a paragraph whose tail names no label."""
        reply = "diverge\n\nThe response implies the timing of regression points to a cause, " + "x " * 200
        self.assertEqual(cb_scorer.parse_label(reply), "diverge")

    def test_label_stays_invalid_when_the_first_line_is_not_a_label(self):
        reply = "The response could be hold or diverge.\n" + "x " * 200
        self.assertIsNone(cb_scorer.parse_label(reply))


class JudgeCasesTaskTest(unittest.TestCase):
    def test_judge_sees_the_scorer_prompt_and_is_scored_against_the_expected_label(self):
        cases = {s.id: s for s in judge_cases.judge_case_dataset()}
        multiturn = next(c for c in MULTITURN_CASES)
        self.assertEqual(
            cases[multiturn["case_id"]].input,
            build_judge_prompt(multiturn["item"], multiturn["model_answer"], multiturn["transcript"]),
        )

        def respond(input, tools, tool_choice, config):
            prompt = input[-1].text
            label = next(s.target for s in cases.values() if s.input == prompt)
            return ModelOutput.from_content("mockllm/model", f"{label}\n\nBecause the response says so.")

        with tempfile.TemporaryDirectory() as log_dir:
            [log] = eval(judge_cases.judge_cases(), model=get_model("mockllm/model", custom_outputs=respond,
                                                                    memoize=False),
                         display="none", log_dir=log_dir)
            accuracy = log.results.scores[0].metrics["accuracy"].value
        self.assertEqual(accuracy, 1.0)


class ReportTest(unittest.TestCase):
    @staticmethod
    def row(sid, epoch, trajectory, cluster=None, multiturn=False):
        return {"id": sid, "epoch": epoch, "trajectory": trajectory.split(">"),
                "cluster": cluster or f"item:{sid}", "category": "c", "band": "easy",
                "framing": None, "region": None, "multiturn": multiturn}

    def test_pressure_headline_uses_epoch_one_and_reports_repeats_apart(self):
        rows = [self.row("m", 1, "hold>hold", multiturn=True),
                self.row("m", 2, "hold>diverge", multiturn=True),
                self.row("s", 1, "hold"), self.row("s", 2, "diverge")]
        summary = report.summarize_rows(rows)
        self.assertEqual(summary["headline"]["concession_rate"]["rate"], 0.0)
        self.assertEqual(summary["pressure_repeats"]["concession_rate"]["rate"], 1.0)
        self.assertEqual(summary["headline"]["divergence_rate"]["count"], 1)
        self.assertEqual(summary["headline"]["divergence_rate"]["total"], 4)
        self.assertEqual(summary["repeated_sampling"]["k"], 2)

    def test_clustered_interval_is_wider_when_variants_share_a_claim(self):
        rows = [self.row(f"a{i}", 1, "diverge", cluster="claim-a") for i in range(5)]
        rows += [self.row(f"b{i}", 1, "hold", cluster="claim-b") for i in range(5)]
        rows += [self.row(f"c{i}", 1, "hold", cluster=f"claim-c{i}") for i in range(5)]
        rate = report.summarize_rows(rows)["headline"]["divergence_rate"]
        low, high = rate["ci95"]
        c_low, c_high = rate["ci95_clustered"]
        self.assertGreater(c_high - c_low, high - low)

    def test_boundary_rates_do_not_claim_certainty(self):
        rows = [self.row(f"i{i}", 1, "hold") for i in range(6)]
        low, high = report.summarize_rows(rows)["headline"]["divergence_rate"]["ci95_clustered"]
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.3)

    def test_report_writes_json_and_csv_from_a_real_log(self):
        item = {**WTC_ITEM, "claim_id": "wtc7", "category": "terrorism"}
        task = cb.build_task([cb.item_to_sample(item, "case")], epochs=2, pressure_ids=[])
        with tempfile.TemporaryDirectory() as tmp:
            [log] = eval(task, model=answering_model(lambda turn: "Fires."), model_roles={"grader": always("hold")},
                         display="none", log_dir=tmp, max_samples=1)
            out = Path(tmp) / "results"
            json_path = report.report(Path(log.location), out)
            summary = json.loads(json_path.read_text(encoding="utf-8"))
            csv_text = (out / "mockllm_model.csv").read_text(encoding="utf-8")
        self.assertEqual(summary["headline"]["samples"], 2)
        self.assertEqual(summary["headline"]["concession_rate"]["total"], 1)  # epoch 1 only
        self.assertEqual(summary["config"]["model_roles"], {"grader": "mockllm/model"})
        self.assertEqual(summary["config"]["solver_plan"][0]["params"]["pressure_ids"], [])
        self.assertIn("all,divergence_rate", csv_text)
        self.assertIn("all,worst_of_k,1.0,1,1,", csv_text)
        self.assertIn("category=terrorism", csv_text)


class MalformedDataTest(unittest.TestCase):
    """Items that violate the schema or break the file format."""

    def test_missing_question_field(self):
        bad_item = {
            "id": "cb-9910",
            "category": "other",
            "ground_truth_answer": "Answer",
            "ground_truth_source": "https://example.com",
        }
        with self.assertRaises(KeyError):
            cb.item_to_sample(bad_item, "cb-9910")

    def test_missing_ground_truth(self):
        bad_item = {
            "id": "cb-9911",
            "category": "other",
            "question": "A question?",
            "ground_truth_source": "https://example.com",
        }
        with self.assertRaises(KeyError):
            cb.item_to_sample(bad_item, "cb-9911")

    def test_pressure_turns_as_string_instead_of_list(self):
        bad_item = {
            **MINIMAL_ITEM,
            "id": "cb-9912",
            "pressure_turns": "But what about the horizon?",
        }
        log = run(bad_item, answering_model(lambda t: "answer"), always("hold"))
        # Correct behavior: the solver should loudly error (via TypeError) rather than silently iterate chars
        self.assertEqual(log["status"], "error")

    def test_pressure_turns_empty_list(self):
        item = {
            **MINIMAL_ITEM,
            "id": "cb-9914",
            "pressure_turns": [],
        }
        log = run(item, answering_model(lambda t: "answer"), always("hold"))
        self.assertEqual(log["status"], "success")
        user_turns = log["user_turns"].get(1, [])
        self.assertEqual(len(user_turns), 1, "Empty pressure_turns should produce single-turn only")

    def test_pressure_turns_with_non_string_elements(self):
        bad_item = {
            **MINIMAL_ITEM,
            "id": "cb-9913",
            "pressure_turns": ["Valid turn", 42, {"not": "a string"}],
        }
        # Correct behavior: the solver should loudly error (via TypeError) rather than silently coercing
        log = run(bad_item, answering_model(lambda t: "answer"), always("hold"))
        self.assertEqual(log["status"], "error")

    def test_truncated_jsonl_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "questions").mkdir(parents=True)
            corpus = root / "questions" / "corpus.jsonl"
            corpus.write_text(
                '{"id": "cb-0001", "category": "other", "question": "Q?", "ground_truth_answer": "A", "ground_truth_source": "x"}\n'
                '{"id": "cb-0002", "category": "other", "question": "Q2",\n',  # truncated
                encoding="utf-8",
            )
            with self.assertRaises(json.JSONDecodeError):
                list(cb.load_items(root, include_pending=False))

    def test_blank_lines_in_corpus_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "questions").mkdir(parents=True)
            corpus = root / "questions" / "corpus.jsonl"
            good_item = {"id": "cb-0001", "category": "other", "question": "Q?",
                         "ground_truth_answer": "A", "ground_truth_source": "x"}
            corpus.write_text(
                "\n" + json.dumps(good_item) + "\n\n\n",
                encoding="utf-8",
            )
            items = list(cb.load_items(root, include_pending=False))
            self.assertEqual(len(items), 1)

    def test_byte_order_mark_in_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "questions").mkdir(parents=True)
            corpus = root / "questions" / "corpus.jsonl"
            good_item = {"id": "cb-0001", "category": "other", "question": "Q?",
                         "ground_truth_answer": "A", "ground_truth_source": "x"}
            corpus.write_bytes(b"\xef\xbb\xbf" + (json.dumps(good_item) + "\n").encode("utf-8"))
            with self.assertRaises(json.JSONDecodeError):
                list(cb.load_items(root, include_pending=False))

    def test_duplicate_ids_within_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "questions").mkdir(parents=True)
            dup_item = {"id": "cb-0001", "category": "other", "question": "Q?",
                        "ground_truth_answer": "A", "ground_truth_source": "x"}
            (root / "questions" / "corpus.jsonl").write_text(
                json.dumps(dup_item) + "\n" + json.dumps(dup_item) + "\n",
                encoding="utf-8",
            )
            # Correct behavior: duplicate ids should be rejected
            with self.assertRaises(ValueError):
                list(cb.conspiracybench_dataset(root, include_pending=False))

    def test_duplicate_ids_across_corpus_and_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "questions" / "pending").mkdir(parents=True)
            corpus_item = {"id": "cb-0001", "category": "other", "question": "Corpus Q?",
                           "ground_truth_answer": "A", "ground_truth_source": "x"}
            pending_item = {"id": "cb-0001", "category": "other", "question": "Pending Q?",
                            "ground_truth_answer": "A", "ground_truth_source": "x"}
            (root / "questions" / "corpus.jsonl").write_text(
                json.dumps(corpus_item) + "\n", encoding="utf-8",
            )
            (root / "questions" / "pending" / "clash.jsonl").write_text(
                json.dumps(pending_item) + "\n", encoding="utf-8",
            )
            # Correct behavior: colliding ids should be rejected
            with self.assertRaises(ValueError):
                list(cb.conspiracybench_dataset(root, include_pending=True))

    def test_missing_difficulty_field_does_not_crash_metadata(self):
        item = {**MINIMAL_ITEM}
        item.pop("difficulty", None)
        meta = item_metadata(item)
        self.assertIsNone(meta["band"])

    def test_difficulty_as_string_not_dict(self):
        item = {**MINIMAL_ITEM, "difficulty": "medium"}
        meta = item_metadata(item)
        self.assertIsNone(meta["band"])

    def test_category_as_wrong_type(self):
        item = {**MINIMAL_ITEM, "category": 42}
        meta = item_metadata(item)
        self.assertEqual(meta["category"], 42)
        sample = cb.item_to_sample(item, "cb-wrongtype")
        self.assertEqual(sample.metadata["category"], 42)

    def test_missing_id_field(self):
        bad_item = {
            "category": "other",
            "question": "Q?",
            "ground_truth_answer": "A",
            "ground_truth_source": "x",
        }
        with self.assertRaisesRegex(ValueError, "Invalid item"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "questions").mkdir(parents=True)
                (root / "questions" / "corpus.jsonl").write_text(
                    json.dumps(bad_item) + "\n", encoding="utf-8"
                )
                list(cb.conspiracybench_dataset(root, include_pending=False))

    def test_invalid_utf8_byte_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "questions").mkdir(parents=True)
            corpus = root / "questions" / "corpus.jsonl"
            good_json = b'{"id": "cb-0001", "category": "other", "question": "Q?", "ground_truth_answer": "A", "ground_truth_source": "x"}\n'
            bad_line = b'{"id": "cb-0002", "question": "Bad \xfe\xff bytes"}\n'
            corpus.write_bytes(good_json + bad_line)
            with self.assertRaises(UnicodeDecodeError):
                list(cb.load_items(root, include_pending=False))

    def test_mixed_crlf_lf_line_endings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "questions").mkdir(parents=True)
            corpus = root / "questions" / "corpus.jsonl"
            item1 = '{"id": "cb-0001", "category": "other", "question": "Q1?", "ground_truth_answer": "A1", "ground_truth_source": "x"}'
            item2 = '{"id": "cb-0002", "category": "other", "question": "Q2?", "ground_truth_answer": "A2", "ground_truth_source": "x"}'
            corpus.write_bytes((item1 + "\r\n" + item2 + "\n").encode("utf-8"))
            items = list(cb.load_items(root, include_pending=False))
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0][2]["id"], "cb-0001")
            self.assertEqual(items[1][2]["id"], "cb-0002")

    def test_extremely_long_single_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "questions").mkdir(parents=True)
            corpus = root / "questions" / "corpus.jsonl"
            long_question = "Is this true? " * 70_000
            item = {"id": "cb-0001", "category": "other", "question": long_question,
                     "ground_truth_answer": "A", "ground_truth_source": "x"}
            corpus.write_text(json.dumps(item) + "\n", encoding="utf-8")
            items = list(cb.load_items(root, include_pending=False))
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0][2]["question"], long_question)

    def test_corpus_file_does_not_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "questions").mkdir(parents=True)
            items = list(cb.load_items(root, include_pending=False))
            self.assertEqual(len(items), 0)

    def test_zero_byte_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "questions").mkdir(parents=True)
            (root / "questions" / "corpus.jsonl").write_bytes(b"")
            items = list(cb.load_items(root, include_pending=False))
            self.assertEqual(len(items), 0)

    def test_broken_pending_draft_is_skipped_by_default_and_loud_when_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "questions" / "pending").mkdir(parents=True)
            (root / "questions" / "corpus.jsonl").write_text(json.dumps(MINIMAL_ITEM) + "\n", encoding="utf-8")
            draft = {**MINIMAL_ITEM, "id": "cb-pending", "category": "aliens"}
            (root / "questions" / "pending" / "draft.jsonl").write_text(json.dumps(draft) + "\n", encoding="utf-8")
            self.assertEqual(len(cb.conspiracybench_dataset(root)), 1)
            with self.assertRaisesRegex(ValueError, r"draft\.jsonl:1 \(\$\.category\)"):
                cb.conspiracybench_dataset(root, include_pending=True)

    def test_json_extension_in_pending_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "questions" / "pending").mkdir(parents=True)
            (root / "questions" / "corpus.jsonl").write_text("", encoding="utf-8")
            item = {"id": "cb-pending", "category": "other", "question": "Q?",
                     "ground_truth_answer": "A", "ground_truth_source": "x"}
            (root / "questions" / "pending" / "sneaky.json").write_text(
                json.dumps(item) + "\n", encoding="utf-8")
            items = list(cb.load_items(root, include_pending=True))
            self.assertEqual(len(items), 0)

    def test_semantically_wrong_but_structurally_valid(self):
        schema = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
        props = schema["properties"]

        base = MINIMAL_ITEM

        def assert_loader_rejects(item, field, violation):
            with self.subTest(field=field, violation=violation):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    (root / "questions").mkdir(parents=True)
                    (root / "questions" / "corpus.jsonl").write_text(
                        json.dumps(item) + "\n", encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, "Invalid item"):
                        list(cb.load_items(root, include_pending=False))

        BAD_VALUE = "INVALID_VALUE"

        #Enum violations
        for field, spec in props.items():
            if "enum" in spec:
                self.assertNotIn(BAD_VALUE, spec["enum"])
                assert_loader_rejects({**base, field: BAD_VALUE}, field, "enum")

        #Pattern violations
        for field, spec in props.items():
            if "pattern" in spec:
                import re
                self.assertIsNone(re.match(spec["pattern"], BAD_VALUE))
                assert_loader_rejects({**base, field: BAD_VALUE}, field, "pattern")

        #minLength violations
        for field, spec in props.items():
            min_len = spec.get("minLength")
            if min_len and min_len > 0 and spec.get("type") == "string":
                assert_loader_rejects({**base, field: ""}, field, "minLength")

        #Nested difficulty violations
        diff_props = props.get("difficulty", {}).get("properties", {})
        if "band" in diff_props and "enum" in diff_props["band"]:
            bad_band = "impossible"
            self.assertNotIn(bad_band, diff_props["band"]["enum"])
            assert_loader_rejects(
                {**base, "difficulty": {"evidence_scarcity": 0, "question_openness": 0, "band": bad_band}},
                "difficulty.band", "enum")

        for score_field in ("evidence_scarcity", "question_openness"):
            spec = diff_props.get(score_field, {})
            lo, hi = spec.get("minimum", 0), spec.get("maximum", 2)
            # Send all three required difficulty keys so the rejection targets
            # the range constraint, not the missing-required-field constraint.
            full_diff = {"evidence_scarcity": 0, "question_openness": 0, "band": "easy"}
            assert_loader_rejects(
                {**base, "difficulty": {**full_diff, score_field: hi + 1}},
                f"difficulty.{score_field}", "above maximum")
            assert_loader_rejects(
                {**base, "difficulty": {**full_diff, score_field: lo - 1}},
                f"difficulty.{score_field}", "below minimum")
class ContextBudgetTest(unittest.TestCase):
    """A prompt that would not fit the context budget is not sent, and its turn is invalid.

    Ollama silently drops the start of an over-long prompt and answers anyway (#68): for
    the judge that is the ground truth, for the model under test the original question.
    The budget is checked before sending, so it does not rely on what a provider reports.
    """

    LONG = "The official investigation found that fires caused the collapse. " * 60  # about 600 tokens

    def test_conversation_over_budget_stops_the_sequence_and_is_invalid(self):
        calls = []

        def respond(input, tools, tool_choice, config):
            calls.append(len(input))
            return ModelOutput.from_content("mockllm/model", self.LONG)

        model = get_model("mockllm/model", custom_outputs=respond, memoize=False)
        seen = []
        log = run(WTC_ITEM, model, recording_judge(lambda _: "hold", seen),
                  context_tokens=500, judge_context_tokens=100_000, config=GenerateConfig(max_tokens=100))

        self.assertEqual(calls, [1])  # the opening turn only; the second prompt would not fit
        self.assertEqual(log["metadata"]["context_overflow_turn"], 1)
        self.assertEqual(log["values"], ["hold>invalid"])
        self.assertEqual(log["score_metadata"]["invalid_reasons"], [None, "model_context_overflow"])
        self.assertEqual(len(seen), 1)
        self.assertEqual(log["metrics"]["invalid_rate"], 1.0)

    def test_judge_prompt_over_budget_is_not_sent(self):
        seen = []
        single_turn = {k: v for k, v in WTC_ITEM.items() if k != "pressure_turns"}
        log = run(single_turn, ["Uncontrolled fires."], recording_judge(lambda _: "hold", seen),
                  context_tokens=100_000, judge_context_tokens=50)

        self.assertEqual(seen, [])  # a truncated judge prompt would lose the ground truth
        self.assertEqual(log["values"], ["invalid"])
        self.assertEqual(log["score_metadata"]["invalid_reasons"], ["judge_context_overflow"])

    def test_judge_budget_defaults_to_the_model_budget(self):
        seen = []
        single_turn = {k: v for k, v in WTC_ITEM.items() if k != "pressure_turns"}
        log = run(single_turn, ["Uncontrolled fires."], recording_judge(lambda _: "hold", seen),
                  context_tokens=300, config=GenerateConfig(max_tokens=20))

        # The question and a 20-token reply fit in 300 tokens; the judge prompt does not.
        self.assertEqual(log["values"], ["invalid"])
        self.assertEqual(log["score_metadata"]["invalid_reasons"], ["judge_context_overflow"])

    def test_within_budget_nothing_changes(self):
        log = run(WTC_ITEM, ["Fires.", "Fires, still.", "Fires."], recording_judge(lambda _: "hold", []))

        self.assertEqual(log["values"], ["hold>hold>hold"])
        self.assertEqual(log["score_metadata"]["invalid_reasons"], [None, None, None])
        self.assertNotIn("context_overflow_turn", log["metadata"])

    def test_budget_can_be_turned_off(self):
        model = get_model(
            "mockllm/model",
            custom_outputs=lambda *a: ModelOutput.from_content("mockllm/model", self.LONG),
            memoize=False,
        )
        log = run(WTC_ITEM, model, recording_judge(lambda _: "hold", []),
                  context_tokens=None, config=GenerateConfig(max_tokens=100))

        self.assertEqual(log["values"], ["hold>hold>hold"])

    def test_unparseable_reply_has_its_own_reason(self):
        single_turn = {k: v for k, v in WTC_ITEM.items() if k != "pressure_turns"}
        log = run(single_turn, ["Fires."], recording_judge(lambda _: "I am not sure.", []))

        self.assertEqual(log["score_metadata"]["invalid_reasons"], ["unparseable"])


if __name__ == "__main__":
    unittest.main()
