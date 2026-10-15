"""Regression test: valid question text that a line splitter or a log writer chokes on.

- U+2028, U+2029 and U+0085 are legal unescaped inside a JSON string, but str.splitlines()
  breaks lines at them, so a valid question was rejected as two broken lines.
- A lone surrogate ("\\ud800", usually half an emoji) is valid JSON and passes schema.json,
  but is not text; an eval run dies writing its log. validate_questions.py must catch it.

Run from the repo root:

    python -m unittest scripts/test_text_edge_cases.py
"""
import ast
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ITEM = {
    "id": "cb-pending",
    "category": "other",
    "question": "placeholder",
    "ground_truth_answer": "A",
    "ground_truth_source": "https://example.com",
}


def run_scripts(pending_bytes: bytes, scripts):
    """Run each script in a scratch copy of the repo whose pending/ holds only pending_bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        shutil.copytree(ROOT / "scripts", repo / "scripts")
        shutil.copytree(ROOT / "questions", repo / "questions")
        shutil.copy(ROOT / "schema.json", repo / "schema.json")
        for f in (repo / "questions" / "pending").glob("*.jsonl"):
            f.unlink()
        (repo / "questions" / "pending" / "t.jsonl").write_bytes(pending_bytes)
        results = {}
        for script in scripts:
            results[script] = subprocess.run(
                [sys.executable, str(repo / "scripts" / script)],
                cwd=repo, capture_output=True, text=True, encoding="utf-8",
            )
        # Bytes, not read_text(): text mode would turn CRLF into LF and hide it.
        corpus = (repo / "questions" / "corpus.jsonl").read_bytes().decode("utf-8")
        return results, corpus


class TextEdgeCaseTest(unittest.TestCase):
    def test_no_script_splits_a_file_with_splitlines(self):
        """Static check, which also covers check_duplicates.py (it needs fastembed to run)."""
        found = []
        for path in sorted((ROOT / "scripts").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "splitlines"
                    and isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Attribute)
                    and node.func.value.func.attr == "read_text"
                ):
                    found.append(f"{path.name}:{node.lineno}")
        self.assertEqual(found, [], "read_text(...).splitlines(): use .split('\\n')")

    def test_separator_inside_a_string_is_one_valid_question(self):
        for name, char in (("U+2028", " "), ("U+2029", " "), ("U+0085", "\u0085")):
            with self.subTest(name):
                question = f"first half{char}second half?"
                line = json.dumps({**ITEM, "question": question}, ensure_ascii=False).encode("utf-8")
                results, corpus = run_scripts(line + b"\n", ("validate_questions.py", "assign_ids.py"))
                for script, run in results.items():
                    self.assertEqual(run.returncode, 0, f"{script}: {run.stdout[-300:]}{run.stderr[-300:]}")
                stored = [json.loads(l) for l in corpus.split("\n") if l.strip()][-1]
                self.assertEqual(stored["question"], question)

    def test_assign_ids_writes_lf_line_endings(self):
        # Text mode writes CRLF on Windows; corpus.jsonl is committed with LF.
        line = json.dumps({**ITEM, "question": "plain?"}).encode("ascii")
        results, corpus = run_scripts(line + b"\n", ("assign_ids.py",))
        self.assertEqual(results["assign_ids.py"].returncode, 0)
        self.assertFalse("\r" in corpus, "assign_ids.py wrote CRLF line endings")

    def test_a_question_of_only_whitespace_is_reported_by_the_validator(self):
        line = json.dumps({**ITEM, "question": "   "}).encode("ascii")
        results, _ = run_scripts(line + b"\n", ("validate_questions.py",))
        run = results["validate_questions.py"]
        self.assertEqual(run.returncode, 1)
        self.assertIn("pending/t.jsonl:1", run.stdout)

    def test_lone_surrogate_is_reported_by_the_validator(self):
        line = json.dumps({**ITEM, "question": "half an emoji \ud83d"}).encode("ascii")
        results, _ = run_scripts(line + b"\n", ("validate_questions.py",))
        run = results["validate_questions.py"]
        self.assertEqual(run.returncode, 1)
        self.assertIn("pending/t.jsonl:1: contains a lone surrogate", run.stdout)


if __name__ == "__main__":
    unittest.main()
