"""Model-free checks for Experiment 4 data, scoring, windows, metrics, and failures."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment import (
    QUESTION,
    QUESTION_ID,
    classify_yes,
    kev_yes_probability,
    laya_window_bounds,
    laya_yes_probability,
    load_verified_train,
    make_inference_payload,
    run_documents,
    shuffled_documents,
)
from report import summarize_attempt


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ProtocolTests(unittest.TestCase):
    def test_payload_contains_only_exported_text_and_fixed_question(self):
        payload = make_inference_payload("assistant answer text")
        self.assertEqual(set(payload), {"state", "questions"})
        self.assertEqual(payload["state"], "assistant answer text")
        self.assertEqual(payload["questions"], {
            QUESTION_ID: {"type": "noul", "instructions": QUESTION},
        })
        self.assertNotIn("label", json.dumps(payload))
        self.assertNotIn("case_id", json.dumps(payload))

    def test_runner_does_not_send_id_label_or_metadata(self):
        seen = []
        document = {"case_id": "case-a", "text": "answer", "label": 1, "metadata": {"attack_family": "hidden"}}

        def infer(payload):
            seen.append(payload)
            return {"probabilities": {"false": 0.5, "true": 0.5}, "window_count": 1, "windows": []}

        rows, status = run_documents("kev", [document], infer)
        self.assertEqual(status, "complete")
        self.assertEqual(rows[0]["predicted_label"], 1)
        self.assertEqual(seen, [make_inference_payload("answer")])

    def test_verified_train_rows_are_the_only_inference_inputs(self):
        documents, release = load_verified_train()
        source_text = {row["case_id"]: row["text"] for row in documents}
        sent = []

        def infer(payload):
            sent.append(payload)
            return {
                "probabilities": {"false": 0.5, "true": 0.5},
                "window_count": 1,
                "windows": [{"index": 0, "token_start": 0, "token_end": 1, "yes_probability": 0.5}],
            }

        rows, status = run_documents("kev", shuffled_documents(documents), infer)
        self.assertEqual(status, "complete")
        self.assertEqual(len(rows), 1960)
        self.assertEqual(len(sent), 1960)
        self.assertEqual({row["case_id"] for row in rows}, set(source_text))
        self.assertEqual(release["evidence"]["requested_rows"], 1960)
        for row, payload in zip(rows, sent):
            self.assertEqual(payload["state"], source_text[row["case_id"]])
            self.assertEqual(payload, make_inference_payload(source_text[row["case_id"]]))

    def test_binary_probability_mapping_and_half_boundary(self):
        self.assertEqual(kev_yes_probability({"false": 0.51, "true": 0.49}), 0.49)
        self.assertEqual(laya_yes_probability({"type": "noul", "noul": 0.5}), 0.5)
        self.assertEqual(classify_yes(0.5), 1)
        self.assertEqual(classify_yes(0.4999), 0)
        with self.assertRaises(ValueError):
            kev_yes_probability({"yes": 0.5, "no": 0.5})

    def test_laya_windows_overlap_and_cover_the_tail(self):
        bounds = laya_window_bounds(1000, 256, 128)
        self.assertEqual(bounds[0], (0, 256))
        self.assertEqual(bounds[-1][1], 1000)
        self.assertTrue(all(right[0] < left[1] for left, right in zip(bounds, bounds[1:])))
        self.assertTrue(all(right[0] <= left[1] for left, right in zip(bounds, bounds[1:])))

    def test_laya_score_must_equal_maximum_window_probability(self):
        documents = [{"case_id": "case-a", "text": "answer", "label": 1}]
        raw = {
            "answer": {"type": "noul", "noul": 0.8},
            "window_count": 2,
            "windows": [
                {"index": 0, "token_start": 0, "token_end": 200, "yes_probability": 0.2},
                {"index": 1, "token_start": 100, "token_end": 250, "yes_probability": 0.8},
            ],
        }
        rows, status = run_documents("laya", documents, lambda _payload: raw)
        self.assertEqual(status, "complete")
        self.assertEqual(rows[0]["yes_probability"], 0.8)
        self.assertEqual(len(rows[0]["windows"]), 2)


class ReportTests(unittest.TestCase):
    def test_known_predictions_reproduce_metrics_and_error_ids(self):
        documents = [
            {"case_id": "id-1", "label": 0},
            {"case_id": "id-2", "label": 0},
            {"case_id": "id-3", "label": 1},
            {"case_id": "id-4", "label": 1},
            {"case_id": "id-5", "label": 1},
            {"case_id": "id-6", "label": 0},
        ]
        metadata = {
            "id-1": {"attack_family": None, "pair_subtype": "ordinary"},
            "id-2": {"attack_family": None, "pair_subtype": "hard_negative"},
            "id-3": {"attack_family": "family_a", "pair_subtype": "ordinary"},
            "id-4": {"attack_family": "family_a", "pair_subtype": "ordinary"},
            "id-5": {"attack_family": "family_b", "pair_subtype": "ordinary"},
            "id-6": {"attack_family": None, "pair_subtype": "hard_negative"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            write_json(attempt / "manifest.json", {
                "model": "kev",
                "attempt_id": "known-fixture",
                "completion_status": "incomplete_memory_failure",
                "phase": "scoring",
            })
            write_jsonl(attempt / "results.jsonl", [
                {"case_id": "id-1", "status": "success", "predicted_label": 0, "latency_seconds": 1.0},
                {"case_id": "id-2", "status": "success", "predicted_label": 1, "latency_seconds": 2.0},
                {"case_id": "id-3", "status": "success", "predicted_label": 0, "latency_seconds": 3.0},
                {"case_id": "id-4", "status": "success", "predicted_label": 1, "latency_seconds": 4.0},
                {"case_id": "id-5", "status": "inference_failed", "error_type": "MemoryError"},
            ])
            summary = summarize_attempt(attempt, documents, metadata)
        self.assertEqual(summary["completion_status"], "incomplete_memory_failure")
        self.assertEqual(summary["metrics"]["confusion_matrix_successful_only"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(summary["metrics"]["successful_accuracy"], 0.5)
        self.assertAlmostEqual(summary["metrics"]["coverage"], 4 / 6)
        self.assertAlmostEqual(summary["metrics"]["accuracy_all_requested"], 2 / 6)
        self.assertEqual(summary["hard_negative_false_positive_ids"], ["id-2"])
        self.assertEqual(summary["misclassified_ids"], ["id-2", "id-3"])
        self.assertEqual(summary["inference_error_ids"], ["id-5"])
        self.assertEqual(summary["not_attempted_ids"], ["id-6"])
        self.assertAlmostEqual(summary["recall_by_attack_family"]["family_a"]["correct_over_all_attacks"], 0.5)
        self.assertEqual(summary["recall_by_attack_family"]["family_b"]["successful_recall"], None)


class MockFullRunTests(unittest.TestCase):
    def test_two_mock_passes_write_exactly_3920_document_records(self):
        documents = [
            {"case_id": f"case-{index:04d}", "text": f"synthetic exported text {index}", "label": index % 2}
            for index in range(1960)
        ]
        ordered = {model: shuffled_documents(documents) for model in ("kev", "laya")}
        self.assertEqual([row["case_id"] for row in ordered["kev"]], [row["case_id"] for row in ordered["laya"]])

        with tempfile.TemporaryDirectory() as temporary:
            record_count = 0
            for model in ("kev", "laya"):
                path = Path(temporary) / f"{model}.jsonl"
                path.touch()

                def infer(_payload, selected=model):
                    if selected == "kev":
                        return {
                            "probabilities": {"false": 0.25, "true": 0.75},
                            "window_count": 1,
                            "windows": [{"index": 0, "token_start": 0, "token_end": 4, "yes_probability": 0.75}],
                        }
                    return {
                        "answer": {"type": "noul", "noul": 0.75},
                        "window_count": 1,
                        "windows": [{"index": 0, "token_start": 0, "token_end": 4, "yes_probability": 0.75}],
                    }

                def save(row, destination=path):
                    with destination.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row) + "\n")

                results, status = run_documents(model, ordered[model], infer, on_result=save)
                self.assertEqual(status, "complete")
                self.assertEqual(len(results), 1960)
                record_count += len(path.read_text(encoding="utf-8").splitlines())
            self.assertEqual(record_count, 3920)

    def test_memory_failure_stops_the_pass_and_marks_it_incomplete(self):
        documents = [{"case_id": f"id-{i}", "text": "text", "label": 0} for i in range(4)]

        def fail(_payload):
            raise MemoryError("synthetic memory allocation failure")

        results, status = run_documents("kev", documents, fail)
        self.assertEqual(status, "incomplete_memory_failure")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "inference_failed")


if __name__ == "__main__":
    unittest.main()
