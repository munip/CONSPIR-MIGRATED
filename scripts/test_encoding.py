"""Regression test: scripts/ must read and write question files as UTF-8.

Without an explicit encoding, Windows falls back to cp1252: a Hindi or curly-quote
question crashes validate_questions.py, and a French one is read as garbage and then
written into corpus.jsonl by assign_ids.py. Run from the repo root:

    python -m unittest scripts/test_encoding.py
"""
import ast
import os
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Text that decodes to garbage or crashes under cp1252: Hindi, French, curly quotes, em dash.
QUESTION = "क्या “Café” isn’t real — या नहीं?"
ITEM = {
    "id": "cb-pending",
    "category": "other",
    "question": QUESTION,
    "ground_truth_answer": "A",
    "ground_truth_source": "https://example.com",
}
# Makes any read/write without encoding= an error on every OS, not just Windows.
STRICT = [sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning"]
# An ASCII-only console, forced so the result does not depend on the machine running the test.
ASCII_CONSOLE = {**os.environ, "PYTHONIOENCODING": "ascii"}


class ScriptEncodingTest(unittest.TestCase):
    def test_no_file_call_without_encoding(self):
        """Static check over every script, including ones the subprocess test can't run."""
        missing = []
        for path in sorted((ROOT / "scripts").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("read_text", "write_text")
                    and "encoding" not in {kw.arg for kw in node.keywords}
                ):
                    missing.append(f"{path.name}:{node.lineno}")
        self.assertEqual(missing, [], "read_text/write_text without encoding=")

    def test_non_ascii_pending_item_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            shutil.copytree(ROOT / "scripts", repo / "scripts")
            shutil.copytree(ROOT / "questions", repo / "questions")
            shutil.copy(ROOT / "schema.json", repo / "schema.json")
            for f in (repo / "questions" / "pending").glob("*.jsonl"):
                f.unlink()
            (repo / "questions" / "pending" / "t.jsonl").write_text(
                json.dumps(ITEM, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            for script in ("validate_questions.py", "assign_ids.py"):
                run = subprocess.run(
                    [*STRICT, str(repo / "scripts" / script)],
                    cwd=repo, capture_output=True, text=True, encoding="utf-8", env=ASCII_CONSOLE,
                )
                self.assertEqual(run.returncode, 0, f"{script}: {run.stderr[-400:]}")
            corpus = (repo / "questions" / "corpus.jsonl").read_text(encoding="utf-8")
            stored = [json.loads(line) for line in corpus.splitlines() if line.strip()][-1]
            self.assertEqual(stored["question"], QUESTION)


if __name__ == "__main__":
    unittest.main()
