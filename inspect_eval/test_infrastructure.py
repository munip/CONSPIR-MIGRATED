#!/usr/bin/env python3
"""Failure-injection tests for the Inspect solver and scorer. Needs inspect-ai; makes no API calls.

    python -m unittest discover -s inspect_eval -p "test_*.py"

mockllm never retries (its ``should_retry`` is False), so an error raised inside it ends the sample
and says nothing about Inspect's own retry path. The providers below say "retry me" for transient
errors and wait zero seconds, so a retry costs nothing. Their answers depend only on the last user
message, so two runs can be compared transcript by transcript.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import anyio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))

from needs_tokenizer import needs_tokenizer
from inspect_ai import eval, eval_retry
from inspect_ai.model import GenerateConfig, ModelAPI, ModelOutput, RetryDecision, modelapi
from tenacity import wait_none

import conspiracybench as cb
import report

# Inspect resolves a task file with a glob, which rejects an absolute path, so give it one relative to where we run.
TASK_FILE = os.path.relpath(HERE / "conspiracybench.py")


class Transient(Exception):
    """Stands in for an HTTP 5xx or a dropped connection."""


class RateLimited(Exception):
    """Stands in for an HTTP 429."""


# What each call was sent, the config it got, and what the next call should do.
STATE = {"calls": [], "judge_calls": [], "model_configs": [], "judge_configs": [], "plan": None, "judge_plan": None}


def reset(plan=None, judge_plan=None):
    STATE.update(calls=[], judge_calls=[], model_configs=[], judge_configs=[], plan=plan, judge_plan=judge_plan)


def _retry_decision(ex):
    if isinstance(ex, RateLimited):
        return RetryDecision.rate_limit()
    if isinstance(ex, (Transient, TimeoutError)):
        return RetryDecision.transient()
    return False


async def _act(action):
    kind, argument = action
    if kind == "raise":
        raise argument
    if kind == "sleep":
        await anyio.sleep(argument)


@modelapi(name="cbflaky")
class FlakyModel(ModelAPI):
    """The model under test: fails or stalls on the calls the test chooses."""

    def __init__(self, model_name, base_url=None, api_key=None, config=GenerateConfig(), **model_args):
        super().__init__(model_name, base_url, api_key, [], config)

    async def generate(self, input, tools, tool_choice, config):
        number = len(STATE["calls"]) + 1
        STATE["calls"].append([(m.role, m.text) for m in input])
        STATE["model_configs"].append((config.max_retries, config.timeout))
        if STATE["plan"]:
            action = STATE["plan"](number, input)
            if action:
                await _act(action)
        last_user = [m.text for m in input if m.role == "user"][-1]
        return ModelOutput.from_content(self.model_name, "re: " + last_user)

    def should_retry(self, ex):
        return _retry_decision(ex)

    def retry_wait(self):
        return wait_none()


@modelapi(name="cbjudge")
class FlakyJudge(ModelAPI):
    """A judge that always says hold, and can fail like the model."""

    def __init__(self, model_name, base_url=None, api_key=None, config=GenerateConfig(), **model_args):
        super().__init__(model_name, base_url, api_key, [], config)

    async def generate(self, input, tools, tool_choice, config):
        number = len(STATE["judge_calls"]) + 1
        STATE["judge_calls"].append([(m.role, m.text) for m in input])
        STATE["judge_configs"].append((config.max_retries, config.timeout))
        if STATE["judge_plan"]:
            action = STATE["judge_plan"](number, input)
            if action:
                await _act(action)
        return ModelOutput.from_content(self.model_name, "hold")

    def should_retry(self, ex):
        return _retry_decision(ex)

    def retry_wait(self):
        return wait_none()


def _item(number, turns=None):
    item = {
        "id": f"cb-990{number}",
        "category": "other",
        "question": f"Question {number}?",
        "ground_truth_answer": "A.",
        "ground_truth_source": "https://example.com",
    }
    if turns:
        item["pressure_turns"] = turns
    return item


MULTI_TURN = _item(1, ["push one", "push two"])
SINGLE_TURN = _item(2)


def run_items(items, plan=None, judge_plan=None, task_kwargs=None, **eval_kwargs):
    """Run the items with the flaky providers and read what the tests inspect while the log exists."""
    reset(plan, judge_plan)
    task = cb.build_task([cb.item_to_sample(i, i["id"]) for i in items], **(task_kwargs or {}))
    with tempfile.TemporaryDirectory() as log_dir:
        [log] = eval(task, model="cbflaky/model", model_roles={"grader": "cbjudge/model"}, display="none",
                     log_dir=log_dir, max_samples=1, **eval_kwargs)
        summary = json.loads(report.report(Path(log.location), Path(log_dir) / "out").read_text(encoding="utf-8"))
        samples = {str(s.id): s for s in log.samples or []}
        return {
            "status": log.status,
            "calls": list(STATE["calls"]),
            "judge_calls": list(STATE["judge_calls"]),
            "model_configs": list(STATE["model_configs"]),
            "judge_configs": list(STATE["judge_configs"]),
            "scores": {i: s.scores["conspiracy_judge"].value for i, s in samples.items() if s.scores},
            "user_turns": {i: [m.text for m in s.messages if m.role == "user"] for i, s in samples.items()},
            "assistant_turns": {i: [m.text for m in s.messages if m.role == "assistant"] for i, s in samples.items()},
            "errors": {i: bool(s.error) for i, s in samples.items()},
            "failed_attempts": {
                i: [e for e in s.events if e.event == "model" and e.model.startswith("cbflaky") and e.error]
                for i, s in samples.items()
            },
            "coverage": summary["coverage"],
        }


class RetryBehaviourTest(unittest.TestCase):
    """What Inspect does when a model call fails. These document guarantees the solver relies on."""

    PROMPTS = [["Question 1?"], ["Question 1?", "push one"], ["Question 1?", "push one"],
               ["Question 1?", "push one"], ["Question 1?", "push one", "push two"]]

    @staticmethod
    def user_turns(calls):
        return [[text for role, text in call if role == "user"] for call in calls]

    @needs_tokenizer
    def test_a_retried_call_sends_the_same_prompt_and_keeps_the_transcript_clean(self):
        result = run_items([MULTI_TURN], plan=lambda n, m: ("raise", Transient("503")) if n in (2, 3) else None)

        self.assertEqual(result["status"], "success")
        self.assertEqual(self.user_turns(result["calls"]), self.PROMPTS)  # the retries repeat turn 1's prompt
        self.assertEqual(result["user_turns"]["cb-9901"], ["Question 1?", "push one", "push two"])
        self.assertEqual(len(result["assistant_turns"]["cb-9901"]), 3)  # no turn duplicated or skipped
        self.assertEqual(result["scores"]["cb-9901"], "hold>hold>hold")

    @needs_tokenizer
    def test_failed_attempts_are_kept_in_the_log_rather_than_dropped(self):
        result = run_items([MULTI_TURN], plan=lambda n, m: ("raise", Transient("503 upstream down")) if n in (2, 3) else None)

        failed = result["failed_attempts"]["cb-9901"]
        self.assertEqual(len(failed), 2)
        self.assertTrue(all("503 upstream down" in str(e.error) for e in failed))

    @needs_tokenizer
    def test_a_rate_limit_is_retried_like_any_transient_error(self):
        result = run_items([MULTI_TURN], plan=lambda n, m: ("raise", RateLimited("429")) if n == 2 else None)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["assistant_turns"]["cb-9901"]), 3)
        self.assertEqual(result["scores"]["cb-9901"], "hold>hold>hold")

    @needs_tokenizer
    def test_retries_that_run_out_are_a_recorded_sample_error(self):
        result = run_items([MULTI_TURN], plan=lambda n, m: ("raise", Transient("503")) if n >= 2 else None,
                           task_kwargs={"config": GenerateConfig(max_retries=2)})

        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["calls"]), 4)  # turn 1, then the failing turn tried 1 + 2 times
        self.assertTrue(result["errors"]["cb-9901"])
        self.assertEqual(result["coverage"]["errored"], 1)

    @needs_tokenizer
    def test_a_slow_attempt_is_abandoned_and_retried(self):
        result = run_items([MULTI_TURN], plan=lambda n, m: ("sleep", 3) if n == 2 else None,
                           task_kwargs={"config": GenerateConfig(attempt_timeout=1)})

        self.assertEqual(result["status"], "success")
        self.assertEqual(self.user_turns(result["calls"]), self.PROMPTS[:2] + self.PROMPTS[3:])
        self.assertEqual(result["user_turns"]["cb-9901"], ["Question 1?", "push one", "push two"])
        self.assertEqual(result["scores"]["cb-9901"], "hold>hold>hold")

    def test_a_stall_that_never_ends_stops_at_the_timeout_with_a_recorded_error(self):
        result = run_items([MULTI_TURN], plan=lambda n, m: ("sleep", 3) if n >= 2 else None,
                           task_kwargs={"config": GenerateConfig(attempt_timeout=1, timeout=2)})

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["errors"]["cb-9901"])


def snapshot(log):
    """What a run produced, sample by sample."""
    return {
        (str(s.id), s.epoch): (
            tuple((m.role, m.text) for m in s.messages if m.role in ("user", "assistant")),
            s.scores["conspiracy_judge"].value,
            (s.metadata or {}).get("turns_planned"),
        )
        for s in log.samples
    }


def run_real_task(plan=None, max_samples=None, **eval_kwargs):
    """The real task over the real corpus (one epoch), with the flaky providers."""
    reset(plan)
    with tempfile.TemporaryDirectory() as log_dir:
        [log] = eval(TASK_FILE, model="cbflaky/model", model_roles={"grader": "cbjudge/model"}, display="none",
                     log_dir=log_dir, task_args={"epochs": 1}, max_samples=max_samples, **eval_kwargs)
        return log, snapshot(log), list(STATE["calls"]), list(STATE["judge_calls"])


@needs_tokenizer
class ResumeAndReproducibilityTest(unittest.TestCase):
    def test_a_resumed_run_matches_an_uninterrupted_one(self):
        clean_log, clean, clean_calls, _ = run_real_task()

        # The endpoint dies whenever a conversation reaches its third user turn, so some samples never finish.
        dies = lambda n, messages: ("raise", Transient("503")) if sum(m.role == "user" for m in messages) >= 3 else None  # noqa: E731
        with tempfile.TemporaryDirectory() as log_dir:
            reset(dies)
            [broken] = eval(TASK_FILE, model="cbflaky/model", model_roles={"grader": "cbjudge/model"},
                            display="none", log_dir=log_dir, task_args={"epochs": 1}, max_retries=2)
            self.assertEqual(broken.status, "error")
            finished = sum(1 for s in broken.samples if not s.error and s.scores)

            reset()  # the endpoint recovers
            [resumed] = eval_retry(broken, log_dir=log_dir)
            resumed_snapshot, retry_calls = snapshot(resumed), list(STATE["calls"])

        self.assertEqual(resumed.status, "success")
        self.assertEqual(resumed_snapshot, clean)  # same samples, transcripts, scores and planned turns
        self.assertEqual((resumed.results.total_samples, resumed.results.completed_samples),
                         (clean_log.results.total_samples, clean_log.results.completed_samples))
        self.assertGreater(finished, 0)
        self.assertLess(len(retry_calls), len(clean_calls))  # samples that had finished were not run again

    def test_the_same_configuration_sends_identical_prompts(self):
        log_one, first, model_one, judge_one = run_real_task(max_samples=1)
        log_two, second, model_two, judge_two = run_real_task(max_samples=20)

        # Concurrency changes the order calls arrive in, not what is sent.
        self.assertEqual(sorted(model_one), sorted(model_two))
        self.assertEqual(sorted(judge_one), sorted(judge_two))
        self.assertEqual(first, second)

    def test_the_log_records_what_is_needed_to_rerun_it(self):
        log, _, _, _ = run_real_task(max_samples=5)

        spec = log.eval
        self.assertEqual(spec.task_args["epochs"], 1)
        for key in ("include_pending", "pressure_subset_per_cell", "subset_seed", "max_tokens", "context_tokens"):
            self.assertIn(key, spec.task_args)
        self.assertEqual(spec.model, "cbflaky/model")
        self.assertEqual({role: m.model for role, m in spec.model_roles.items()}, {"grader": "cbjudge/model"})
        if spec.revision is not None:  # only recorded inside a git checkout
            self.assertIsNotNone(spec.revision.commit)  # the code that ran
        self.assertGreater(len(spec.dataset.sample_ids), 0)  # the items that ran
        params = {s.solver: s.params for s in log.plan.steps}
        self.assertIsNone(params["pressure_turns"]["pressure_ids"])  # every conversation, every epoch


class RetryDefaultsTest(unittest.TestCase):
    """A dead endpoint must end in an error, not retry forever.

    Inspect retries a failing call for as long as the provider says it can be retried, unless
    max_retries or timeout is set. The task sets both, for the model under test and for the judge.
    """

    def test_the_task_sets_bounded_retries_by_default(self):
        task = cb.build_task([cb.item_to_sample(SINGLE_TURN, "x")])

        self.assertEqual(task.config.max_retries, cb.DEFAULT_MAX_RETRIES)
        self.assertEqual(task.config.timeout, cb.DEFAULT_TIMEOUT)
        self.assertIsNotNone(cb.DEFAULT_MAX_RETRIES)
        self.assertIsNotNone(cb.DEFAULT_TIMEOUT)

    def test_the_registered_task_takes_them_as_arguments_with_the_same_defaults(self):
        self.assertEqual((cb.conspiracybench().config.max_retries, cb.conspiracybench().config.timeout),
                         (cb.DEFAULT_MAX_RETRIES, cb.DEFAULT_TIMEOUT))
        tuned = cb.conspiracybench(max_retries=3, timeout=60)
        self.assertEqual((tuned.config.max_retries, tuned.config.timeout), (3, 60))

    def test_a_caller_supplied_config_wins_over_the_default(self):
        task = cb.build_task([cb.item_to_sample(SINGLE_TURN, "x")], config=GenerateConfig(max_retries=3))

        self.assertEqual(task.config.max_retries, 3)
        self.assertEqual(task.config.timeout, cb.DEFAULT_TIMEOUT)

    @needs_tokenizer
    def test_the_defaults_reach_both_the_model_and_the_judge(self):
        result = run_items([SINGLE_TURN])

        expected = (cb.DEFAULT_MAX_RETRIES, cb.DEFAULT_TIMEOUT)
        self.assertEqual(set(result["model_configs"]), {expected})
        self.assertEqual(set(result["judge_configs"]), {expected})

    @needs_tokenizer
    def test_a_dead_model_endpoint_ends_in_an_error(self):
        result = run_items([SINGLE_TURN], plan=lambda n, m: ("raise", Transient("503")), task_kwargs={"max_retries": 2})

        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["calls"]), 3)  # one try and two retries

    @needs_tokenizer
    def test_a_dead_judge_ends_in_an_error(self):
        result = run_items([SINGLE_TURN], judge_plan=lambda n, m: ("raise", Transient("503")), task_kwargs={"max_retries": 2})

        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["judge_calls"]), 3)

    def test_none_on_the_command_line_turns_a_limit_off(self):
        # -T name=none reaches the task as the string "none", which must not end up in the config.
        task = cb.conspiracybench(max_retries="none", timeout="None")

        self.assertIsNone(task.config.max_retries)
        self.assertIsNone(task.config.timeout)

    def test_numbers_given_as_text_become_numbers(self):
        task = cb.conspiracybench(max_retries="3", timeout="600")

        self.assertEqual((task.config.max_retries, task.config.timeout), (3, 600))

    def test_a_value_that_is_not_a_number_fails_before_the_run_not_in_its_log(self):
        with self.assertRaises(ValueError):
            cb.conspiracybench(max_retries="often")

    def test_an_empty_value_fails_instead_of_removing_the_limit(self):
        # -T max_retries= (nothing after the =) used to mean "no limit", silently.
        for value in ("", "  "):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cb.conspiracybench(max_retries=value)

    def test_the_context_budget_can_be_turned_off_from_the_command_line(self):
        # The README says context_tokens=none turns the budget off; from the command line it was a string.
        reset()
        task = cb.conspiracybench(context_tokens="none")
        with tempfile.TemporaryDirectory() as log_dir:
            [log] = eval(task, model="cbflaky/model", model_roles={"grader": "cbjudge/model"},
                         display="none", log_dir=log_dir, limit=1)
            status = log.status

        self.assertEqual(status, "success")

    @needs_tokenizer
    def test_a_setting_on_the_command_line_beats_the_default(self):
        result = run_items([SINGLE_TURN], plan=lambda n, m: ("raise", Transient("503")), max_retries=4)

        self.assertEqual(len(result["calls"]), 5)  # eval(max_retries=...) is what --max-retries sets


class RootCauseTest(unittest.TestCase):
    """The report says why a sample errored, not only that Inspect gave up retrying."""

    @needs_tokenizer
    def test_exhausted_retries_report_what_kept_failing(self):
        result = run_items([SINGLE_TURN], plan=lambda n, m: ("raise", Transient("503 upstream down")),
                           task_kwargs={"max_retries": 2})

        text = result["coverage"]["errored_samples"][0]["error"]
        self.assertIn("Transient", text)
        self.assertIn("503 upstream down", text)
        self.assertNotIn("Future", text)

    @needs_tokenizer
    def test_attempt_timeouts_report_the_timeout(self):
        result = run_items([SINGLE_TURN], plan=lambda n, m: ("sleep", 3),
                           task_kwargs={"config": GenerateConfig(max_retries=1, attempt_timeout=1)})

        text = result["coverage"]["errored_samples"][0]["error"]
        self.assertIn("AttemptTimeoutError", text)
        self.assertNotIn("Future", text)

    @needs_tokenizer
    def test_a_run_that_stops_reports_why(self):
        # Inspect's default fail_on_error stops the run at the first errored sample. The run's own
        # error is the same unreadable RetryError, so it gets the same treatment.
        result = run_items([SINGLE_TURN, MULTI_TURN], plan=lambda n, m: ("raise", Transient("503 upstream down")),
                           task_kwargs={"max_retries": 1})

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["coverage"]["run_error"], "Transient: 503 upstream down (retries exhausted)")

    @needs_tokenizer
    def test_a_run_that_finishes_has_no_run_error(self):
        self.assertIsNone(run_items([SINGLE_TURN])["coverage"]["run_error"])

    def test_other_errors_keep_their_own_message(self):
        error = SimpleNamespace(message="ConnectionError('connection dropped')", traceback="Traceback (most recent call last):\n")

        self.assertEqual(report.root_cause(error), "ConnectionError('connection dropped')")

    def test_the_cause_is_found_in_a_chained_traceback(self):
        error = SimpleNamespace(
            message="RetryError(<Future at 0x1 state=finished raised Transient>)",
            traceback=(
                "Traceback (most recent call last):\n"
                '  File "x.py", line 1, in call\n'
                "    raise Transient('503 upstream down')\n"
                "pkg.mod.Transient: 503 upstream down\n"
                "\n"
                "The above exception was the direct cause of the following exception:\n"
                "\n"
                "Traceback (most recent call last):\n"
                '  File "y.py", line 2, in retry\n'
                "tenacity.RetryError: RetryError[<Future at 0x1 state=finished raised Transient>]\n"
            ),
        )

        self.assertEqual(report.root_cause(error), "Transient: 503 upstream down (retries exhausted)")

    def test_a_retry_error_without_a_readable_cause_falls_back_to_its_message(self):
        error = SimpleNamespace(message="RetryError(<Future at 0x1 state=finished raised Transient>)", traceback=None)

        self.assertTrue(report.root_cause(error).startswith("RetryError"))


if __name__ == "__main__":
    unittest.main()
