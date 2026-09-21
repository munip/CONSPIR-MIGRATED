#!/usr/bin/env python3
"""Tests for eval/score_outcomes.py. Run from the repo root:

    python -m unittest discover -s eval -p "test_*.py"
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_outcomes import summarize_outcomes, summarize_trajectories


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


if __name__ == "__main__":
    unittest.main()
