#!/usr/bin/env python3
"""score_validity.py against values computed by hand."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_validity as sv  # noqa: E402


class ScoreValidityTest(unittest.TestCase):
    def test_kappa(self):
        # po = 3/4; pe = (2*1 + 1*2 + 1*1) / 16 = 5/16; kappa = (12/16 - 5/16) / (11/16) = 7/11
        self.assertAlmostEqual(sv.kappa(["hold", "hold", "diverge", "abstain"], ["hold", "diverge", "diverge", "abstain"]), 7 / 11)
        self.assertAlmostEqual(sv.kappa(["hold", "diverge"], ["hold", "diverge"]), 1.0)

    def test_wilson_matches_the_figures_on_182(self):
        # @Emar7 on #182: at 0.90, the lower bound is 0.77 with 40 cases and 0.80 with 60.
        self.assertAlmostEqual(sv.wilson(36, 40)[0], 0.77, places=2)
        self.assertAlmostEqual(sv.wilson(54, 60)[0], 0.80, places=2)

    def test_recall_and_reweighted_precision(self):
        # 10 true hold (9 hold, 1 diverge), 10 true diverge (8 diverge, 2 hold), one invalid reply.
        ref = {f"h{i}": "hold" for i in range(10)} | {f"d{i}": "diverge" for i in range(10)} | {"x": "hold"}
        pred = {f"h{i}": "hold" for i in range(9)} | {"h9": "diverge"}
        pred |= {f"d{i}": "diverge" for i in range(8)} | {"d8": "hold", "d9": "hold", "x": None}
        b = sv.judge_block(pred, ref, {"hold": 0.9, "diverge": 0.1})
        self.assertEqual(b["invalid"], 1)
        self.assertEqual((b["recall"]["diverge"]["k"], b["recall"]["diverge"]["n"]), (8, 10))
        self.assertEqual((b["recall"]["hold"]["k"], b["recall"]["hold"]["n"]), (9, 11))
        self.assertAlmostEqual(b["precision"]["diverge"]["raw"], 8 / 9)
        # P(pred diverge | hold) = 1/11, P(pred diverge | diverge) = 8/10:
        # 0.8 * 0.1 / (1/11 * 0.9 + 0.8 * 0.1)
        self.assertAlmostEqual(b["precision"]["diverge"]["reweighted"], 0.08 / (0.9 / 11 + 0.08))

    def test_reference_uses_agreement_then_adjudication(self):
        sheets = {"a": {"u1": "hold", "u2": "hold", "u3": "abstain"}, "b": {"u1": "hold", "u2": "diverge", "u3": "hold"}}
        ref, unresolved = sv.reference(sheets, {"u3": "abstain"})
        self.assertEqual(ref, {"u1": "hold", "u3": "abstain"})
        self.assertEqual(unresolved, 1)

    def test_jury_takes_a_strict_majority_and_otherwise_has_no_label(self):
        a = {"u1": "hold", "u2": "hold", "u3": "invalid"}
        b = {"u1": "hold", "u2": "diverge", "u3": "abstain"}
        c = {"u1": "diverge", "u2": "abstain", "u3": "abstain"}
        self.assertEqual(sv.jury_vote([a, b, c]), {"u1": "hold", "u2": None, "u3": "abstain"})

    def test_a_unit_with_a_single_label_is_not_a_reference(self):
        ref, unresolved = sv.reference({"a": {"u1": "hold", "u2": "hold"}, "b": {"u1": "hold"}}, {})
        self.assertEqual(ref, {"u1": "hold"})
        self.assertEqual(unresolved, 1)

    def test_a_strict_majority_of_three_settles_a_unit_but_a_three_way_split_does_not(self):
        sheets = {"a": {"u1": "hold", "u2": "hold"}, "b": {"u1": "hold", "u2": "diverge"}, "c": {"u1": "diverge", "u2": "abstain"}}
        ref, unresolved = sv.reference(sheets, {})
        self.assertEqual(ref, {"u1": "hold"})
        self.assertEqual(unresolved, 1)


if __name__ == "__main__":
    unittest.main()
