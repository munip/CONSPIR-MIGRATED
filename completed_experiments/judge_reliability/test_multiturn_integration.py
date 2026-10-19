import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "eval"))

import run_baseline

class MultiturnRunnerIntegrationTest(unittest.TestCase):
    def test_default_fixture_contract(self):
        case_path = run_baseline.find_case_file(REPO, None)
        cases = run_baseline.load_cases(case_path)
        self.assertEqual(len(cases), 15)
        run_baseline.validate_cases(cases, is_default=True)
        self.assertEqual(run_baseline.detect_run_kind(cases), "single_turn_regression")

    def test_multiturn_fixture_contract(self):
        mt_path = REPO / "eval" / "judge_cases_multiturn.jsonl"
        cases = run_baseline.load_cases(mt_path)
        self.assertEqual(len(cases), 11)
        run_baseline.validate_cases(cases, is_default=False)
        self.assertEqual(run_baseline.detect_run_kind(cases), "multi_turn_regression")

    def test_validation_rejects_bad_labels(self):
        bad_cases = [{"case_id": "c-1", "expected_label": "invalid_label", "model_answer": "a", "item": {"question": "q"}}]
        with self.assertRaises(SystemExit):
            run_baseline.validate_cases(bad_cases, is_default=False)

    def test_validation_rejects_duplicate_ids(self):
        dup_cases = [
            {"case_id": "dup-1", "expected_label": "hold", "model_answer": "a", "item": {"question": "q"}},
            {"case_id": "dup-1", "expected_label": "diverge", "model_answer": "b", "item": {"question": "q2"}}
        ]
        with self.assertRaises(SystemExit):
            run_baseline.validate_cases(dup_cases, is_default=False)

    def test_output_dir_isolation_and_no_self_archiving(self):
        mt_path = REPO / "eval" / "judge_cases_multiturn.jsonl"
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_out = Path(tmp_dir) / "custom_results"
            sys_argv_backup = sys.argv
            try:
                sys.argv = [
                    "run_baseline.py",
                    "--repo", str(REPO),
                    "--case-file", str(mt_path),
                    "--output-dir", str(custom_out),
                    "--model", "custom/test-model",
                    "--inspect-only"
                ]
                exit_code = run_baseline.main()
                self.assertEqual(exit_code, 0)

                self.assertTrue((custom_out / "baseline_preflight.json").exists())
                zip_path = custom_out / "judge_reliability_preflight.zip"
                self.assertTrue(zip_path.exists())

                pf_data = json.loads((custom_out / "baseline_preflight.json").read_text(encoding="utf-8"))
                self.assertEqual(pf_data["model"], "custom/test-model")
                self.assertEqual(pf_data["case_count"], 11)
                self.assertEqual(pf_data["run_kind"], "multi_turn_regression")

                with zipfile.ZipFile(zip_path, "r") as z:
                    names = z.namelist()
                    self.assertNotIn("judge_reliability_preflight.zip", names)
                    self.assertNotIn(str(zip_path), names)

                default_dir = REPO / "judge_reliability_results"
                self.assertFalse(default_dir.exists())
            finally:
                sys.argv = sys_argv_backup

    def test_external_fixture_snapshot_and_privacy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ext_case = Path(tmp_dir) / "external_fixture.jsonl"
            ext_case.write_text(
                json.dumps({
                    "case_id": "ext-1",
                    "expected_label": "hold",
                    "model_answer": "ans",
                    "item": {"question": "q", "ground_truth_answer": ""}
                }) + "\n",
                encoding="utf-8"
            )
            custom_out = Path(tmp_dir) / "out"
            sys_argv_backup = sys.argv
            try:
                sys.argv = [
                    "run_baseline.py",
                    "--repo", str(REPO),
                    "--case-file", str(ext_case),
                    "--output-dir", str(custom_out),
                    "--inspect-only"
                ]
                exit_code = run_baseline.main()
                self.assertEqual(exit_code, 0)

                # Check archive has safe snapshot
                zip_path = custom_out / "judge_reliability_preflight.zip"
                with zipfile.ZipFile(zip_path, "r") as z:
                    self.assertIn("snapshot/external/external_fixture.jsonl", z.namelist())

                # Check preflight metadata leaks no local host path
                pf_data = json.loads((custom_out / "baseline_preflight.json").read_text(encoding="utf-8"))
                self.assertEqual(pf_data["case_file"], "external/external_fixture.jsonl")
                self.assertNotIn("Users", pf_data["case_file"])
                self.assertNotIn(":", pf_data["case_file"])
            finally:
                sys.argv = sys_argv_backup

    def test_resume_offline_and_missing_progress_rejection(self):
        mt_path = REPO / "eval" / "judge_cases_multiturn.jsonl"
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "results"
            out_dir.mkdir(parents=True)
            (out_dir / "baseline_results_partial.jsonl").write_text(
                json.dumps({"case_id": "mt-001", "parsed_label": "hold"}) + "\n",
                encoding="utf-8"
            )

            # Ensure GROQ_API_KEY is not in environment
            old_key = os.environ.pop("GROQ_API_KEY", None)
            sys_argv_backup = sys.argv
            try:
                sys.argv = [
                    "run_baseline.py",
                    "--repo", str(REPO),
                    "--case-file", str(mt_path),
                    "--output-dir", str(out_dir),
                    "--resume"
                ]
                with self.assertRaises(SystemExit) as ctx:
                    run_baseline.main()
                self.assertIn("progress metadata file", str(ctx.exception))
            finally:
                sys.argv = sys_argv_backup
                if old_key is not None:
                    os.environ["GROQ_API_KEY"] = old_key

    def test_resume_missing_field_rejection_and_preflight_preservation(self):
        mt_path = REPO / "eval" / "judge_cases_multiturn.jsonl"
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "results"
            out_dir.mkdir(parents=True)
            (out_dir / "baseline_results_partial.jsonl").write_text(
                json.dumps({"case_id": "mt-001", "parsed_label": "hold"}) + "\n",
                encoding="utf-8"
            )
            # Progress file missing 'run_kind'
            (out_dir / "baseline_progress.json").write_text(
                json.dumps({
                    "case_file_sha256": run_baseline.sha256_file(mt_path),
                    "prompt_file_sha256": run_baseline.sha256_file(REPO / "eval" / "judge_prompt.py"),
                    "model": run_baseline.DEFAULT_MODEL
                }),
                encoding="utf-8"
            )
            # Pre-existing preflight file
            old_pf_text = '{"original": "do_not_overwrite"}'
            (out_dir / "baseline_preflight.json").write_text(old_pf_text, encoding="utf-8")

            old_key = os.environ.pop("GROQ_API_KEY", None)
            sys_argv_backup = sys.argv
            try:
                sys.argv = [
                    "run_baseline.py",
                    "--repo", str(REPO),
                    "--case-file", str(mt_path),
                    "--output-dir", str(out_dir),
                    "--resume"
                ]
                with self.assertRaises(SystemExit) as ctx:
                    run_baseline.main()
                self.assertIn("missing required provenance field 'run_kind'", str(ctx.exception))

                # Verify old preflight file was not overwritten
                self.assertEqual((out_dir / "baseline_preflight.json").read_text(encoding="utf-8"), old_pf_text)
            finally:
                sys.argv = sys_argv_backup
                if old_key is not None:
                    os.environ["GROQ_API_KEY"] = old_key

    def test_case_file_without_output_dir_rejected(self):
        mt_path = REPO / "eval" / "judge_cases_multiturn.jsonl"
        old_key = os.environ.pop("GROQ_API_KEY", None)
        sys_argv_backup = sys.argv
        try:
            sys.argv = [
                "run_baseline.py",
                "--repo", str(REPO),
                "--case-file", str(mt_path),
                "--inspect-only"
            ]
            with self.assertRaises(SystemExit) as ctx:
                run_baseline.main()
            self.assertIn("--output-dir is required when --case-file is used", str(ctx.exception))
        finally:
            sys.argv = sys_argv_backup
            if old_key is not None:
                os.environ["GROQ_API_KEY"] = old_key

    def test_case_file_matching_default_output_dir_rejected(self):
        mt_path = REPO / 'eval' / 'judge_cases_multiturn.jsonl'
        old_key = os.environ.pop('GROQ_API_KEY', None)
        sys_argv_backup = sys.argv
        try:
            sys.argv = [
                'run_baseline.py',
                '--repo', str(REPO),
                '--case-file', str(mt_path),
                '--output-dir', 'judge_reliability_results',
                '--inspect-only',
            ]
            with self.assertRaises(SystemExit) as ctx:
                run_baseline.main()
            self.assertIn('different from the default judge_reliability_results directory', str(ctx.exception))
        finally:
            sys.argv = sys_argv_backup
            if old_key is not None:
                os.environ['GROQ_API_KEY'] = old_key

    def test_prompt_transcript_fidelity_and_label_isolation(self):
        mt_path = REPO / "eval" / "judge_cases_multiturn.jsonl"
        cases = run_baseline.load_cases(mt_path)
        prompt_fn = run_baseline.load_prompt_module(REPO / "eval" / "judge_prompt.py")

        cases_to_test = [c for c in cases if c.get("case_id") in ("mt-001", "mt-003", "mt-004", "mt-006", "mt-010")]
        self.assertEqual(len(cases_to_test), 5)

        for c in cases_to_test:
            prompt = run_baseline.build_prompt_adapter(prompt_fn, c)
            self.assertIn(c["model_answer"], prompt)
            last_idx = -1
            for turn in c["transcript"]:
                idx = prompt.find(turn["content"])
                self.assertNotEqual(idx, -1, f"Missing turn content: {turn['content']}")
                self.assertGreater(idx, last_idx, "Transcript turns are out of chronological order in prompt")
                last_idx = idx
            self.assertIn("User:", prompt)
            for label in ("hold", "diverge", "abstain"):
                self.assertNotIn(f"expected={label}", prompt)
                self.assertNotIn(f"expected_label: {label}", prompt)

if __name__ == "__main__":
    unittest.main()
