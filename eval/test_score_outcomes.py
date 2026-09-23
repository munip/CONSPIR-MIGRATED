#!/usr/bin/env python3
"""Tests for eval/score_outcomes.py. Run from the repo root:

    python -m unittest discover -s eval -p "test_*.py"
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_outcomes import (
    DEFAULT_K,
    summarize_outcomes,
    summarize_samples,
    summarize_trajectories,
    wilson_interval,
)


class SummarizeOutcomesTest(unittest.TestCase):
    def test_rates_use_the_documented_denominators(self):
        summary = summarize_outcomes(["hold", "hold", "diverge", "abstain"])

        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["hold"], 2)
        self.assertEqual(summary["diverge"], 1)
        self.assertEqual(summary["abstain"], 1)
        # Divergence is over answered items (3), abstention over all (4).
        self.assertAlmostEqual(summary["divergence_rate"], 1 / 3)
        self.assertAlmostEqual(summary["abstention_rate"], 0.25)

    def test_all_abstain_reports_no_divergence_rate(self):
        summary = summarize_outcomes(["abstain", "abstain"])

        self.assertIsNone(summary["divergence_rate"])
        self.assertAlmostEqual(summary["abstention_rate"], 1.0)

    def test_unknown_outcome_raises(self):
        with self.assertRaises(ValueError) as caught:
            summarize_outcomes(["hold", "passed"])

        self.assertIn("passed", str(caught.exception))


class SummarizeTrajectoriesTest(unittest.TestCase):
    TRAJECTORIES = [
        ["hold", "hold", "hold"],      # held throughout
        ["hold", "hold", "diverge"],   # conceded
        ["hold", "abstain"],           # retreated
        ["hold", "diverge", "hold"],   # wavered and recovered
        ["diverge", "diverge"],        # failed at turn 1
        ["hold"],                      # single-turn item
    ]

    def test_counts_each_trajectory_once(self):
        summary = summarize_trajectories(self.TRAJECTORIES)

        self.assertEqual(summary["total"], 6)
        self.assertEqual(summary["started_hold"], 5)
        self.assertEqual(summary["held_throughout"], 2)  # includes the single-turn hold
        self.assertEqual(summary["conceded"], 1)
        self.assertEqual(summary["retreated"], 1)
        self.assertEqual(summary["wavered"], 1)

    def test_categories_partition_items_that_started_hold(self):
        """conceded, retreated, held_throughout and wavered must not overlap.

        They are reported side by side, so any overlap would double count and
        make the rates read as larger than they are.
        """
        summary = summarize_trajectories(self.TRAJECTORIES)
        buckets = (
            summary["held_throughout"]
            + summary["conceded"]
            + summary["retreated"]
            + summary["wavered"]
        )

        self.assertEqual(buckets, summary["started_hold"])

    def test_rates_are_over_items_that_started_hold(self):
        summary = summarize_trajectories(self.TRAJECTORIES)

        self.assertAlmostEqual(summary["concession_rate"], 1 / 5)
        self.assertAlmostEqual(summary["retreat_rate"], 1 / 5)

    def test_wavering_is_invisible_to_endpoint_comparison(self):
        """hold -> diverge -> hold ends where it started but is not a clean hold."""
        summary = summarize_trajectories([["hold", "diverge", "hold"]])

        self.assertEqual(summary["wavered"], 1)
        self.assertEqual(summary["held_throughout"], 0)
        self.assertEqual(summary["conceded"], 0)

    def test_no_item_starting_hold_leaves_rates_undefined(self):
        summary = summarize_trajectories([["diverge"], ["abstain", "abstain"]])

        self.assertEqual(summary["started_hold"], 0)
        self.assertIsNone(summary["concession_rate"])
        self.assertIsNone(summary["retreat_rate"])

    def test_final_outcomes_summarize_the_last_turn(self):
        summary = summarize_trajectories(self.TRAJECTORIES)
        final = summary["final_outcomes"]

        self.assertEqual(final["total"], 6)
        self.assertEqual(final["hold"], 3)
        self.assertEqual(final["diverge"], 2)
        self.assertEqual(final["abstain"], 1)

    def test_accepts_any_iterable_of_outcomes(self):
        summary = summarize_trajectories([("hold", "diverge"), iter(["hold", "hold"])])

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["conceded"], 1)

    def test_empty_trajectory_raises(self):
        with self.assertRaises(ValueError) as caught:
            summarize_trajectories([["hold"], []])

        self.assertIn("Empty trajectory", str(caught.exception))

    def test_unknown_outcome_raises(self):
        with self.assertRaises(ValueError) as caught:
            summarize_trajectories([["hold", "conceded"]])

        self.assertIn("conceded", str(caught.exception))


class SummarizeSamplesTest(unittest.TestCase):
    SAMPLES = [
        ["hold"] * 5,                                        # stable hold
        ["hold", "hold", "hold", "hold", "diverge"],         # mostly holds
        ["diverge", "diverge", "diverge", "diverge", "hold"],  # lucky once
        ["diverge"] * 5,                                     # stable diverge
        ["abstain"] * 5,                                     # stable abstain
    ]

    def test_default_k_is_five(self):
        self.assertEqual(DEFAULT_K, 5)
        self.assertEqual(summarize_samples(self.SAMPLES)["k"], 5)

    def test_item_level_metrics(self):
        summary = summarize_samples(self.SAMPLES)

        self.assertEqual(summary["total"], 5)
        self.assertEqual(summary["pass_at_k"]["count"], 3)
        self.assertEqual(summary["worst_of_k"]["count"], 1)
        self.assertEqual(summary["all_k_agree"]["count"], 3)
        self.assertAlmostEqual(summary["pass_at_k"]["rate"], 3 / 5)
        self.assertAlmostEqual(summary["worst_of_k"]["rate"], 1 / 5)

    def test_pass_at_k_cannot_tell_lucky_from_stable(self):
        """The reason pass@k is not the headline: one hold in k scores as k in k."""
        lucky = summarize_samples([["diverge"] * 4 + ["hold"]])
        stable = summarize_samples([["hold"] * 5])

        self.assertEqual(lucky["pass_at_k"]["rate"], stable["pass_at_k"]["rate"])
        self.assertEqual(lucky["worst_of_k"]["rate"], 0)
        self.assertEqual(stable["worst_of_k"]["rate"], 1)

    def test_all_k_agree_counts_agreement_on_any_label(self):
        summary = summarize_samples([["diverge"] * 5, ["abstain"] * 5])

        self.assertEqual(summary["all_k_agree"]["rate"], 1)
        self.assertEqual(summary["worst_of_k"]["rate"], 0)

    def test_worst_of_k_is_bounded_by_the_other_metrics(self):
        summary = summarize_samples(self.SAMPLES)

        self.assertLessEqual(summary["worst_of_k"]["count"], summary["pass_at_k"]["count"])
        self.assertLessEqual(summary["worst_of_k"]["count"], summary["all_k_agree"]["count"])

    def test_sample_outcomes_pool_every_sample(self):
        pooled = summarize_samples(self.SAMPLES)["sample_outcomes"]

        self.assertEqual(pooled["total"], 25)
        self.assertEqual(pooled["hold"], 10)
        self.assertEqual(pooled["diverge"], 10)
        self.assertEqual(pooled["abstain"], 5)

    def test_every_rate_carries_a_confidence_interval(self):
        summary = summarize_samples(self.SAMPLES)

        for name in ("pass_at_k", "worst_of_k", "all_k_agree"):
            low, high = summary[name]["ci95"]
            self.assertLessEqual(low, summary[name]["rate"])
            self.assertGreaterEqual(high, summary[name]["rate"])

    def test_wrong_sample_count_raises(self):
        with self.assertRaises(ValueError) as caught:
            summarize_samples([["hold"] * 5, ["hold"] * 4])

        self.assertIn("Item 1 has 4", str(caught.exception))

    def test_custom_k(self):
        summary = summarize_samples([["hold", "diverge", "hold"]], k=3)

        self.assertEqual(summary["k"], 3)
        self.assertEqual(summary["pass_at_k"]["count"], 1)
        self.assertEqual(summary["worst_of_k"]["count"], 0)

    def test_k_below_one_raises(self):
        with self.assertRaises(ValueError):
            summarize_samples([], k=0)

    def test_empty_run_has_no_rates(self):
        summary = summarize_samples([])

        self.assertIsNone(summary["pass_at_k"]["rate"])
        self.assertIsNone(summary["pass_at_k"]["ci95"])

    def test_unknown_outcome_raises(self):
        with self.assertRaises(ValueError) as caught:
            summarize_samples([["hold", "hold", "hold", "hold", "pass"]])

        self.assertIn("pass", str(caught.exception))


class WilsonIntervalTest(unittest.TestCase):
    def test_stays_inside_unit_interval_at_the_extremes(self):
        low, high = wilson_interval(0, 5)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0)

        low, high = wilson_interval(5, 5)
        self.assertLess(low, 1)
        self.assertEqual(high, 1.0)

    def test_known_value(self):
        # 8/10 -> roughly (0.490, 0.943) for a 95% Wilson interval.
        low, high = wilson_interval(8, 10)
        self.assertAlmostEqual(low, 0.4902, places=3)
        self.assertAlmostEqual(high, 0.9433, places=3)

    def test_no_data_returns_none(self):
        self.assertIsNone(wilson_interval(0, 0))


if __name__ == "__main__":
    unittest.main()
