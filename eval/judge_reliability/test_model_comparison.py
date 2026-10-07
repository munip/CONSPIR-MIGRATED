from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_baseline as baseline
import run_model_comparison as comparison


class ModelComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = baseline.load_cases(ROOT / "eval" / "judge_cases.jsonl")
        cls.tasks = comparison.build_tasks(cls.cases)

    def test_standard_suite_has_84_calls(self):
        self.assertEqual(len(self.tasks), 84)

    def test_core_suite_has_60_calls(self):
        core = [t for t in self.tasks if t["phase"] == "core"]
        self.assertEqual(len(core), 60)

    def test_core_has_15_cases_per_family(self):
        core = [t for t in self.tasks if t["phase"] == "core"]

        for family in ("clean", "format", "irrelevant", "injection"):
            with self.subTest(family=family):
                self.assertEqual(
                    sum(t["family"] == family for t in core),
                    15,
                )

    def test_stability_panel_has_12_groups_and_24_extra_calls(self):
        stability = [t for t in self.tasks if t["phase"] == "stability"]
        self.assertEqual(len(stability), 24)

        groups = {
            t["stability_group"]
            for t in self.tasks
            if t["stability_group"] is not None
        }

        self.assertEqual(len(groups), 12)

        for group in groups:
            rows = [
                t
                for t in self.tasks
                if t["stability_group"] == group
            ]
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                {r["repeat_index"] for r in rows},
                {1, 2, 3},
            )

    def test_stability_clean_panel_covers_topics_and_labels(self):
        ids = {
            t["source_case_id"]
            for t in self.tasks
            if t["phase"] == "core"
            and t["family"] == "clean"
            and t["stability_group"] is not None
        }

        self.assertEqual(
            ids,
            {
                "judge-001", "judge-003", "judge-005",
                "judge-006", "judge-008", "judge-010",
                "judge-011", "judge-012", "judge-013",
            },
        )

    def test_wilson_interval_for_perfect_small_sample_is_bounded(self):
        low, high = comparison.wilson_interval(15, 15)
        self.assertGreater(low, 0.75)
        self.assertLessEqual(high, 1.0)

    def test_accuracy_block_counts_invalid_as_wrong(self):
        rows = [
            {"parsed_correct": True, "parsed_label": "hold"},
            {"parsed_correct": False, "parsed_label": None},
        ]

        block = comparison.accuracy_block(rows)

        self.assertEqual(block["correct"], 1)
        self.assertEqual(block["total"], 2)
        self.assertEqual(block["invalid_count"], 1)
        self.assertEqual(block["accuracy"], 0.5)


    def test_invalid_predictions_reduce_macro_f1(self):
        rows = [
            {"expected_label": "hold", "parsed_label": "hold"},
            {"expected_label": "hold", "parsed_label": None},
            {"expected_label": "diverge", "parsed_label": "diverge"},
            {"expected_label": "abstain", "parsed_label": "abstain"},
        ]

        self.assertAlmostEqual(
            comparison.macro_f1_counting_invalid(rows),
            8 / 9,
        )

    def test_three_invalid_outputs_do_not_count_as_stable(self):
        self.assertFalse(
            comparison.valid_three_way_agreement([None, None, None])
        )
        self.assertTrue(
            comparison.valid_three_way_agreement(
                ["hold", "hold", "hold"]
            )
        )

    def test_task_plan_hash_changes_when_plan_changes(self):
        original = comparison.task_plan_sha256(self.tasks)

        changed = [dict(task) for task in self.tasks]
        changed[0]["candidate_answer"] += " changed"

        self.assertNotEqual(
            original,
            comparison.task_plan_sha256(changed),
        )


if __name__ == "__main__":
    unittest.main()
