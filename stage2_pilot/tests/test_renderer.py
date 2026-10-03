import copy
import json
import tempfile
import unittest
from pathlib import Path

from stage2_pilot.renderer import LedgerError, render_entry, render_ledger, write_candidates

ROOT = Path(__file__).resolve().parents[2]


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
        cls.fixture = json.loads((ROOT / "stage2_pilot/fixtures/synthetic_ledger.json").read_text(encoding="utf-8"))

    def entry(self):
        return copy.deepcopy(self.fixture["entries"][0])

    def test_synthetic_fixture_renders_with_real_schema(self):
        item = render_entry(self.entry(), self.schema, allow_synthetic=True)
        self.assertEqual(item["id"], "cb-pending")
        self.assertEqual(len(item["pressure_turns"]), 1)
        self.assertNotIn("verification_status", item)
        self.assertNotIn("synthetic_fixture", item)

    def test_repeat_render_is_byte_identical_and_newline_terminated(self):
        first = render_ledger(self.fixture, self.schema, allow_synthetic=True)
        second = render_ledger(self.fixture, self.schema, allow_synthetic=True)
        self.assertEqual(first, second)
        self.assertEqual(list(first), ["synthetic-demo.jsonl"])
        self.assertEqual(first["synthetic-demo.jsonl"].count(b"\n"), 1)
        self.assertTrue(first["synthetic-demo.jsonl"].endswith(b"\n"))

    def test_requires_explicit_review(self):
        e = self.entry()
        e["verification_status"] = "draft"
        with self.assertRaises(LedgerError):
            render_entry(e, self.schema, allow_synthetic=True)

    def test_requires_source_mapping_for_every_proposition(self):
        e = self.entry()
        e["factual_propositions"].append("Second synthetic proposition.")
        with self.assertRaisesRegex(LedgerError, "missing source mappings"):
            render_entry(e, self.schema, allow_synthetic=True)

    def test_rejects_more_than_three_pressure_turns(self):
        e = self.entry()
        e["pressure_turns"] = ["a", "b", "c", "d"]
        with self.assertRaises(LedgerError):
            render_entry(e, self.schema, allow_synthetic=True)

    def test_rejects_mismatched_difficulty(self):
        e = self.entry()
        e["difficulty"]["band"] = "hard"
        with self.assertRaises(LedgerError):
            render_entry(e, self.schema, allow_synthetic=True)

    def test_synthetic_export_is_locked_by_default(self):
        with self.assertRaises(LedgerError):
            render_entry(self.entry(), self.schema)

    def test_real_export_requires_inventory(self):
        real = self.entry()
        real["synthetic_fixture"] = False
        real["slug"] = "demo"
        with self.assertRaisesRegex(LedgerError, "open-PR inventory"):
            render_ledger({"version": 1, "entries": [real]}, self.schema)

    def test_rejects_duplicate_claims_in_same_batch(self):
        e = self.entry()
        other = self.entry()
        other["slug"] = "synthetic-demo-two"
        with self.assertRaisesRegex(LedgerError, "Duplicate slug or claim_id"):
            render_ledger({"version": 1, "entries": [e, other]}, self.schema, allow_synthetic=True)

    def test_rejects_claim_in_supplied_inventory(self):
        e = self.entry()
        real = copy.deepcopy(e)
        real["synthetic_fixture"] = False
        real["slug"] = "demo"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "inventory.json"
            p.write_text(json.dumps({"as_of": "2026-09-26", "items": [{
                "source": "open-pr/123", "claim_id": real["claim_id"], "question": "Different wording"
            }]}), encoding="utf-8")
            with self.assertRaisesRegex(LedgerError, "same claim ID"):
                render_ledger({"version": 1, "entries": [real]}, self.schema, inventory=p)

    def test_write_is_non_destructive(self):
        files = render_ledger(self.fixture, self.schema, allow_synthetic=True)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out"
            write_candidates(files, p, ROOT)
            self.assertEqual((p / "synthetic-demo.jsonl").read_bytes(), files["synthetic-demo.jsonl"])
            with self.assertRaisesRegex(LedgerError, "overwrite"):
                write_candidates(files, p, ROOT)

    def test_cannot_write_to_questions_pending(self):
        files = render_ledger(self.fixture, self.schema, allow_synthetic=True)
        with self.assertRaisesRegex(LedgerError, "Never write candidates"):
            write_candidates(files, ROOT / "questions" / "pending", ROOT)


    def test_six_real_drafts_remain_unapproved(self):
        path = ROOT / "stage2_pilot/ledgers/six_question_drafts.json"
        ledger = json.loads(path.read_text(encoding="utf-8"))
        entries = ledger["entries"]

        self.assertEqual(len(entries), 6)
        self.assertTrue(all(
            e["verification_status"] == "draft" for e in entries
        ))
        self.assertEqual(
            sum(bool(e.get("pressure_turns")) for e in entries), 3
        )
        self.assertEqual(
            sum(not e.get("pressure_turns") for e in entries), 3
        )

    def test_six_real_drafts_schema_preview(self):
        path = ROOT / "stage2_pilot/ledgers/six_question_drafts.json"
        ledger = json.loads(path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as d:
            inventory = Path(d) / "inventory.json"
            inventory.write_text(
                json.dumps({
                    "as_of": "2026-09-27",
                    "items": []
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                LedgerError, "source review must be explicitly completed"
            ):
                render_ledger(
                    ledger, self.schema, inventory=inventory
                )

            # In-memory override for a schema test ONLY.
            preview = copy.deepcopy(ledger)

            for entry in preview["entries"]:
                entry["verification_status"] = "source-reviewed"

            rendered = render_ledger(
                preview, self.schema, inventory=inventory
            )

            self.assertEqual(len(rendered), 6)
            self.assertEqual(
                set(rendered),
                {e["slug"] + ".jsonl" for e in preview["entries"]}
            )

            for content in rendered.values():
                self.assertEqual(len(content.splitlines()), 1)
                self.assertEqual(
                    json.loads(content.decode("utf-8"))["id"],
                    "cb-pending",
                )


if __name__ == "__main__":
    unittest.main()
