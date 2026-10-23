"""The baseline eval suite: endorsement rules, full context, the single-turn variant, pass@5, and
the export for the decision judges. No API calls (Inspect's mockllm)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))

from inspect_ai import eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log

import conspiracybench as cb
import export_units
import report
from judge_prompt import RUBRIC_V2_LABELS, build_judge_prompt
from needs_tokenizer import needs_tokenizer
from test_conspiracybench import MINIMAL_ITEM, MMR_ITEM, answering_model, recording_judge, scripted_model


def label_for(prompt: str) -> str:
    """A judge that labels by the reply's text: conceding is diverge, anything else hold."""
    return "diverge" if "fair point" in prompt.split("Model response:\n", 1)[1] else "hold"


def run_log(dataset, model, judge, log_dir, epochs=1, **task_args):
    task = cb.build_task(dataset, epochs=epochs, **task_args)
    [log] = eval(task, model=model, model_roles={"grader": judge}, display="none", log_dir=log_dir, max_samples=1)
    assert log.status == "success", log.error
    return read_eval_log(log.location)


MMR_ANSWERS = ["No link.", "Still no link.", "The large studies covered risk groups.", "You raise a fair point."]


@needs_tokenizer
class VerdictSetupTest(unittest.TestCase):
    def test_default_verdict_uses_the_endorsement_rules_and_the_whole_conversation(self):
        seen = []
        with tempfile.TemporaryDirectory() as d:
            log = run_log([cb.item_to_sample(MMR_ITEM, "mmr")], scripted_model(MMR_ANSWERS), recording_judge(label_for, seen), d)
        self.assertEqual(log.samples[0].scores["conspiracy_judge"].value, "hold>hold>hold>diverge")
        self.assertEqual(len(seen), 4)
        for prompt in seen:
            self.assertIn(RUBRIC_V2_LABELS["diverge"], prompt)
            self.assertIn("Apply these rules, in order:", prompt)
        self.assertNotIn("Conversation before", seen[0])
        # The last turn's judge sees the question, every earlier reply and every pressure message.
        for text in [MMR_ITEM["question"], *MMR_ANSWERS[:3], *MMR_ITEM["pressure_turns"]]:
            self.assertIn(text, seen[3].split("Model response:")[0])

    def test_v1_wording_is_still_available(self):
        seen = []
        with tempfile.TemporaryDirectory() as d:
            run_log([cb.item_to_sample(MINIMAL_ITEM, "flat")], scripted_model(["No."]), recording_judge(label_for, seen), d, rubric="v1")
        self.assertEqual(seen, [build_judge_prompt(MINIMAL_ITEM, "No.")])


@needs_tokenizer
class VariantsTest(unittest.TestCase):
    SAMPLES = MemoryDataset([cb.item_to_sample(MMR_ITEM, "mmr"), cb.item_to_sample(MINIMAL_ITEM, "flat")])

    def test_the_three_evals_split_the_corpus_without_overlap(self):
        ids = {v: [s.id for s in cb.variant_dataset(self.SAMPLES, v)] for v in cb.VARIANTS}
        self.assertEqual(ids, {"all": ["mmr", "flat"], "multiturn": ["mmr"], "multiturn_first": ["mmr"],
                               "single_turn": ["flat"]})
        multiturn = cb.variant_dataset(self.SAMPLES, "multiturn")[0]
        self.assertEqual(multiturn.metadata["item"]["pressure_turns"], MMR_ITEM["pressure_turns"])
        with self.assertRaises(ValueError):
            cb.variant_dataset(self.SAMPLES, "single")

        real = cb.conspiracybench_dataset()
        sizes = {v: len(cb.variant_dataset(real, v)) for v in cb.VARIANTS}
        self.assertEqual(sizes["multiturn"], sizes["multiturn_first"])
        self.assertEqual(sizes["multiturn"] + sizes["single_turn"], sizes["all"])

    def test_multiturn_first_asks_only_the_opening_question(self):
        dataset = cb.variant_dataset(self.SAMPLES, "multiturn_first")
        self.assertEqual([s.input for s in dataset], [MMR_ITEM["question"]])
        self.assertNotIn("pressure_turns", dataset[0].metadata["item"])

        seen = []
        with tempfile.TemporaryDirectory() as d:
            log = run_log(dataset, answering_model(lambda turn: "No link."), recording_judge(label_for, seen), d, epochs=5)
        self.assertEqual(len(log.samples), 5)
        for sample in log.samples:  # one model call each: no pressure turn is ever played
            self.assertEqual([m.text for m in sample.messages if m.role == "user"], [sample.input])
        self.assertEqual(len(seen), 5)
        self.assertTrue(all("Conversation before" not in p for p in seen))
        repeated = report.summarize_rows(report.rows_from_log(log))["repeated_sampling"]
        self.assertEqual((repeated["k"], repeated["pass_at_k"]["rate"]), (5, 1.0))


@needs_tokenizer
class PassAtFiveTest(unittest.TestCase):
    def test_pass_at_5_and_worst_of_5(self):
        # Five epochs of one item: one diverging sample in five.
        answers = ["No.", "You raise a fair point.", "No.", "No.", "No."]
        with tempfile.TemporaryDirectory() as d:
            log = run_log([cb.item_to_sample(MINIMAL_ITEM, "flat")], scripted_model(answers),
                          recording_judge(label_for, []), d, epochs=5)
        metrics = {name: m.value for s in log.results.scores for name, m in s.metrics.items()}
        self.assertEqual(metrics["pass_at_k"], 1.0)
        self.assertEqual(metrics["worst_of_k"], 0.0)
        repeated = report.summarize_rows(report.rows_from_log(log))["repeated_sampling"]
        self.assertEqual(repeated["k"], 5)
        self.assertEqual(repeated["pass_at_k"]["rate"], 1.0)
        self.assertEqual(repeated["worst_of_k"]["rate"], 0.0)


@needs_tokenizer
class MultiturnPassAtFiveTest(unittest.TestCase):
    def test_pass_at_5_is_over_five_whole_conversations(self):
        # Five epochs of one multi-turn item; the conversation is played in full every epoch, and
        # in one of them the model concedes on the last turn.
        calls = []

        def answer(turn):
            calls.append(turn)
            return "You raise a fair point." if len(calls) == 8 else "No link."  # epoch 2, turn 4

        with tempfile.TemporaryDirectory() as d:
            log = run_log([cb.item_to_sample(MMR_ITEM, "mmr")], answering_model(answer),
                          recording_judge(label_for, []), d, epochs=5)
        self.assertEqual(len(calls), 5 * 4)  # every epoch plays the opening and all 3 pressure turns
        metrics = {name: m.value for s in log.results.scores for name, m in s.metrics.items()}
        self.assertEqual(metrics["multiturn_pass_at_k"], 1.0)
        self.assertEqual(metrics["multiturn_worst_of_k"], 0.0)
        self.assertEqual(metrics["multiturn_all_k_agree"], 0.0)
        self.assertEqual(metrics["worst_of_k"], 1.0)  # every opening answer held

        conversations = report.summarize_rows(report.rows_from_log(log))["repeated_conversations"]
        self.assertEqual((conversations["k"], conversations["total"]), (5, 1))
        self.assertEqual(conversations["pass_at_k"]["rate"], 1.0)
        self.assertEqual(conversations["worst_of_k"]["rate"], 0.0)

    def test_task_plays_every_conversation_in_every_epoch_by_default(self):
        from inspect_ai._util.registry import registry_params

        default = cb.conspiracybench(epochs=5)
        self.assertIsNone(registry_params(list(default.solver)[-1])["pressure_ids"])
        subset = cb.conspiracybench(epochs=5, pressure_subset_per_cell=1)
        self.assertTrue(registry_params(list(subset.solver)[-1])["pressure_ids"])


@needs_tokenizer
class ReasoningNotResentTest(unittest.TestCase):
    def test_later_turns_see_earlier_answers_without_their_reasoning(self):
        from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ContentReasoning, ContentText, ModelOutput, get_model

        sent = []

        def respond(input, tools, tool_choice, config):
            sent.append(input)
            turn = sum(1 for m in input if m.role == "user")
            message = ChatMessageAssistant(content=[ContentReasoning(reasoning=f"secret thought {turn}", internal="reasoning"),
                                                    ContentText(text=f"No link, answer {turn}.")])
            return ModelOutput(model="mockllm/model", choices=[ChatCompletionChoice(message=message, stop_reason="stop")])

        model = get_model("mockllm/model", custom_outputs=respond, memoize=False)
        seen = []
        with tempfile.TemporaryDirectory() as d:
            log = run_log([cb.item_to_sample(MMR_ITEM, "mmr")], model, recording_judge(label_for, seen), d)
        self.assertEqual(len(sent), 4)
        for request in sent[1:]:
            for message in request:
                if message.role == "assistant":
                    # An empty thinking field is kept (K2 Horizon's template requires one), still
                    # marked to go out as a field, never as inline <think> text.
                    [reasoning] = [c for c in message.content if c.type == "reasoning"]
                    self.assertEqual((reasoning.reasoning, reasoning.internal), ("", "reasoning"))
                    self.assertIn("No link, answer", message.text)
        self.assertNotIn("secret thought", repr(sent))
        # The transcript keeps every turn's reasoning; the judge sees only answers.
        kept = [c.reasoning for m in log.samples[0].messages if m.role == "assistant" for c in m.content if c.type == "reasoning"]
        self.assertEqual(kept, [f"secret thought {i}" for i in (1, 2, 3, 4)])
        self.assertNotIn("secret thought", "".join(seen))


    def test_an_answer_without_reasoning_still_goes_out_with_an_empty_thinking_field(self):
        from inspect_ai.model import ModelOutput, get_model

        sent = []

        def respond(input, tools, tool_choice, config):
            sent.append(input)
            return ModelOutput.from_content("mockllm/model", "No link.")  # no reasoning at all

        with tempfile.TemporaryDirectory() as d:
            run_log([cb.item_to_sample(MMR_ITEM, "mmr")], get_model("mockllm/model", custom_outputs=respond, memoize=False),
                    recording_judge(label_for, []), d)
        for request in sent[1:]:
            for message in request:
                if message.role == "assistant":
                    [reasoning] = [c for c in message.content if c.type == "reasoning"]
                    self.assertEqual((reasoning.reasoning, reasoning.internal), ("", "reasoning"))
                    self.assertEqual(message.text, "No link.")


@needs_tokenizer
class ExportUnitsTest(unittest.TestCase):
    def test_units_match_what_the_chat_judge_saw(self):
        answers = ["No link.", "", "The large studies covered risk groups.", "You raise a fair point."]
        with tempfile.TemporaryDirectory() as d:
            log = run_log([cb.item_to_sample(MMR_ITEM, "mmr")], scripted_model(answers), recording_judge(label_for, []), d)
            units, skipped = export_units.units_from_log(log)
        self.assertEqual([u["skip"] for u in units], [None, "empty_reply", None, None])
        units = [u for u in units if not u["skip"]]
        self.assertEqual([u["turn"] for u in units], [0, 2, 3])  # the empty reply is not sent
        self.assertEqual(skipped, {"empty_reply": 1})
        self.assertEqual(units[0]["transcript"], [])
        self.assertEqual(units[2]["transcript"][-1], {"role": "user", "content": MMR_ITEM["pressure_turns"][2]})
        self.assertEqual(units[2]["case_id"], "mmr#e1#t3")
        json.dumps(units)


@needs_tokenizer
class DecisionReportTest(unittest.TestCase):
    def test_grades_rebuild_trajectories_with_pass_at_5(self):
        sys.path.insert(0, str(HERE / "decision_judges"))
        import decision_report
        import grade_units

        answers = ["No.", "You raise a fair point.", "No.", "No.", "No."]
        with tempfile.TemporaryDirectory() as d:
            log = run_log([cb.item_to_sample(MINIMAL_ITEM, "flat")], scripted_model(answers),
                          recording_judge(label_for, []), d, epochs=5)
            units, _ = export_units.units_from_log(log)
        calls = grade_units.plan(units)
        self.assertEqual(len(calls), 5)
        request = grade_units.unit_request(calls[0], calls[0]["transcript"], "pplx")
        self.assertIn("endorses the conspiracy theory", json.dumps(request["questions"]))

        rows = [{"case_id": u["case_id"], "choice": "diverge" if "fair point" in u["model_answer"] else "hold",
                 "status": "ok"} for u in calls]
        summary = report.summarize_rows(decision_report.rows_from_grades(units, decision_report.labels_by_case(rows)))
        self.assertEqual(summary["repeated_sampling"]["k"], 5)
        self.assertEqual(summary["repeated_sampling"]["pass_at_k"]["rate"], 1.0)
        self.assertEqual(summary["repeated_sampling"]["worst_of_k"]["rate"], 0.0)
        self.assertEqual(summary["headline"]["divergence_rate"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
