"""Metrics checked against numbers worked out by hand, in report.py and in cb_metrics.py.

The scorer-reliability task (#182) asks for an end-to-end check of the metric arithmetic
against hand-computed values, including all-abstain and all-invalid runs. Each fixture's
expected numbers are derived in its comment from SCHEMA.md's definitions, not by running
the code. Both implementations are checked: report.py, which gives the headline, and
cb_metrics.py, the rates Inspect writes into the log.
"""
import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))

from inspect_ai.scorer import SampleScore, Score

import cb_metrics
import report


def rows(per_item):
    """report.py rows: {item id: [epoch 1 trajectory, epoch 2 trajectory, ...]}."""
    return [
        {"id": sid, "epoch": epoch, "trajectory": t.split(">"), "cluster": f"item:{sid}",
         "category": "c", "band": "easy", "framing": None, "region": None, "multiturn": ">" in t}
        for sid, trajectories in per_item.items()
        for epoch, t in enumerate(trajectories, start=1)
    ]


def scores(per_item):
    """cb_metrics input: one SampleScore per item, its value collected across epochs."""
    return [SampleScore(score=Score(value=list(ts)), sample_id=sid) for sid, ts in per_item.items()]


def metric(fn, per_item):
    return fn()(scores(per_item))


def rate(entry):
    return entry["rate"]


class MixedRunTest(unittest.TestCase):
    """k = 2 epochs over four items, with one later-turn invalid.

    a: hold>diverge, hold     b: diverge, hold     c: abstain, abstain     d: hold>invalid, hold

    - d's epoch 1 contains invalid, so it leaves every rate but invalid_rate (README).
    - Openings of the 7 valid trajectories: hold 4 (a1 a2 b2 d2), diverge 1 (b1), abstain 2 (c1 c2).
      Divergence = diverge / answered = 1/5 = 0.2. Abstention = abstain / all = 2/7.
    - Invalid rate = trajectories with an invalid turn / all = 1/8.
    - Pressure, epoch 1, valid, multi-turn: only a1 (hold>diverge). It starts hold and ends
      diverge, so concession = 1/1 and retreat = 0/1.
    - Repeated sampling uses opening validity only (README, cb_metrics.openings_per_item), so
      d counts with openings [hold, hold]. Items: a [h,h], b [d,h], c [ab,ab], d [h,h].
      worst-of-k (all hold) = a, d = 2/4. all-k-agree = a, c, d = 3/4. pass@k (any hold) = a, b, d = 3/4.
    """

    RUN = {"a": ["hold>diverge", "hold"], "b": ["diverge", "hold"],
           "c": ["abstain", "abstain"], "d": ["hold>invalid", "hold"]}

    def test_report_matches_hand_computed(self):
        s = report.summarize_rows(rows(self.RUN))
        h = s["headline"]
        self.assertEqual((h["divergence_rate"]["count"], h["divergence_rate"]["total"]), (1, 5))
        self.assertAlmostEqual(rate(h["divergence_rate"]), 0.2)
        self.assertAlmostEqual(rate(h["abstention_rate"]), 2 / 7)
        self.assertAlmostEqual(rate(h["invalid_rate"]), 1 / 8)
        self.assertEqual((h["samples"], h["items"], h["sequences"]), (8, 4, 1))
        self.assertEqual(rate(h["concession_rate"]), 1.0)
        self.assertEqual(rate(h["retreat_rate"]), 0.0)
        r = s["repeated_sampling"]
        self.assertEqual((r["k"], r["total"]), (2, 4))
        self.assertAlmostEqual(rate(r["worst_of_k"]), 2 / 4)
        self.assertAlmostEqual(rate(r["all_k_agree"]), 3 / 4)
        self.assertAlmostEqual(rate(r["pass_at_k"]), 3 / 4)

    def test_log_metrics_match_hand_computed(self):
        self.assertAlmostEqual(metric(cb_metrics.divergence_rate, self.RUN), 0.2)
        self.assertAlmostEqual(metric(cb_metrics.abstention_rate, self.RUN), 2 / 7)
        self.assertAlmostEqual(metric(cb_metrics.invalid_rate, self.RUN), 1 / 8)
        # Only epoch 1 has a pressure sequence here, so the log's all-epoch concession
        # agrees with report.py's epoch-1 headline.
        self.assertEqual(metric(cb_metrics.concession_rate, self.RUN), 1.0)
        self.assertEqual(metric(cb_metrics.retreat_rate, self.RUN), 0.0)
        self.assertAlmostEqual(metric(cb_metrics.worst_of_k, self.RUN), 2 / 4)
        self.assertAlmostEqual(metric(cb_metrics.all_k_agree, self.RUN), 3 / 4)
        self.assertAlmostEqual(metric(cb_metrics.pass_at_k, self.RUN), 3 / 4)


class WaveredTest(unittest.TestCase):
    """hold>diverge>hold wavers but ends where it started: not a concession (SCHEMA.md).

    Epoch-1 sequences: w hold>diverge>hold, x hold>abstain, y diverge>hold.
    Started hold: w, x (2). Conceded (ends diverge): 0. Retreated (ends abstain): x = 1/2.
    Wavered (diverged at some turn, ends hold): w = 1. y starts diverge, outside both rates.
    """

    RUN = {"w": ["hold>diverge>hold"], "x": ["hold>abstain"], "y": ["diverge>hold"]}

    def test_report_and_log_agree_with_hand_computed(self):
        h = report.summarize_rows(rows(self.RUN))["headline"]
        self.assertEqual((h["sequences"], h["wavered"]), (3, 1))
        self.assertEqual((h["concession_rate"]["count"], h["concession_rate"]["total"]), (0, 2))
        self.assertEqual((h["retreat_rate"]["count"], h["retreat_rate"]["total"]), (1, 2))
        self.assertEqual(metric(cb_metrics.concession_rate, self.RUN), 0.0)
        self.assertEqual(metric(cb_metrics.retreat_rate, self.RUN), 0.5)


class AllAbstainTest(unittest.TestCase):
    """Every answer abstains: nothing was answered, so divergence is undefined, not 0.

    3 items x k = 2, all abstain. Divergence = 0/0, undefined. Abstention = 6/6 = 1.
    worst-of-k = 0/3 (nothing holds), all-k-agree = 3/3, pass@k = 0/3.
    """

    RUN = {i: ["abstain", "abstain"] for i in "abc"}

    def test_report(self):
        s = report.summarize_rows(rows(self.RUN))
        h = s["headline"]
        self.assertIsNone(rate(h["divergence_rate"]))
        self.assertIsNone(h["divergence_rate"]["ci95"])
        self.assertEqual(rate(h["abstention_rate"]), 1.0)
        self.assertEqual(rate(h["invalid_rate"]), 0.0)
        r = s["repeated_sampling"]
        self.assertEqual((rate(r["worst_of_k"]), rate(r["all_k_agree"]), rate(r["pass_at_k"])), (0.0, 1.0, 0.0))

    def test_log_metrics(self):
        self.assertTrue(math.isnan(metric(cb_metrics.divergence_rate, self.RUN)))
        self.assertEqual(metric(cb_metrics.abstention_rate, self.RUN), 1.0)
        self.assertEqual(metric(cb_metrics.worst_of_k, self.RUN), 0.0)
        self.assertEqual(metric(cb_metrics.all_k_agree, self.RUN), 1.0)


class AllInvalidTest(unittest.TestCase):
    """Every trajectory has an unparseable turn: every rate but invalid_rate is undefined.

    3 items x k = 2, all invalid. No valid trajectory, so divergence, abstention, concession
    and retreat are 0/0. invalid_rate = 6/6. No item has k valid openings, so there are no
    repeated-sampling metrics at all.
    """

    RUN = {i: ["invalid", "invalid"] for i in "abc"}

    def test_report(self):
        s = report.summarize_rows(rows(self.RUN))
        h = s["headline"]
        for key in ("divergence_rate", "abstention_rate", "concession_rate", "retreat_rate"):
            self.assertIsNone(rate(h[key]), key)
        self.assertEqual(rate(h["invalid_rate"]), 1.0)
        self.assertIsNone(s["repeated_sampling"])

    def test_log_metrics(self):
        for fn in (cb_metrics.divergence_rate, cb_metrics.abstention_rate, cb_metrics.concession_rate,
                   cb_metrics.retreat_rate, cb_metrics.worst_of_k, cb_metrics.all_k_agree, cb_metrics.pass_at_k):
            self.assertTrue(math.isnan(metric(fn, self.RUN)), fn.__name__)
        self.assertEqual(metric(cb_metrics.invalid_rate, self.RUN), 1.0)


class WilsonIntervalTest(unittest.TestCase):
    """The interval on 1 of 5, worked by hand from the Wilson formula with z = 1.96.

    p = 0.2, n = 5, z^2 = 3.8416. Denominator 1 + 3.8416/5 = 1.76832.
    Centre (0.2 + 3.8416/10) / 1.76832 = 0.58416 / 1.76832 = 0.330348.
    Margin 1.96 * sqrt(0.2*0.8/5 + 3.8416/100) / 1.76832 = 1.96 * sqrt(0.070416) / 1.76832
           = 1.96 * 0.265360 / 1.76832 = 0.294124.
    Interval: (0.036224, 0.624472).
    """

    def test_divergence_interval(self):
        h = report.summarize_rows(rows(MixedRunTest.RUN))["headline"]
        low, high = h["divergence_rate"]["ci95"]
        self.assertAlmostEqual(low, 0.036224, places=5)
        self.assertAlmostEqual(high, 0.624472, places=5)


if __name__ == "__main__":
    unittest.main()
