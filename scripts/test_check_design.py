"""The design check that every experiment request has to pass. Run from the repo root:

    python -m unittest scripts/test_check_design.py
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_design import check  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_design.py"

FILLED = """# Does the LoRA cut divergence?

## Claim

If we fine-tune Qwen3-4B on data/final_v4, its divergence rate on ConspiracyBench drops by at least 5 points
compared with the base model.

## Unit and metric

One corpus item (315), k=5 samples each. Divergence rate = diverge verdicts / all verdicts, by pplx-decider.

## Controls

The base model, and a LoRA trained on the same number of shuffled generic chat rows. Always-hold scores 0%.

## Data

Corpus at commit 58de878c2. SFT set filtered (>=40-char replies, no slurs): 43.6k -> 24.4k rows. No SFT row
mentions a corpus claim; nothing from the corpus is trained on.

## Noise

Three seeds, and a bootstrap over claim_id for a 95% interval.

## Readout

A drop under 5 points, or an interval that covers 0, refutes the claim.
"""


class Check(unittest.TestCase):
    def test_a_filled_design_passes(self):
        self.assertEqual(check(FILLED), [])

    def test_the_template_fails_every_stage(self):
        errors = {f["stage"] for f in check((ROOT / "experiment_requests" / "TEMPLATE.md").read_text(encoding="utf-8"))
                  if f["level"] == "error"}
        self.assertEqual(errors, {"claim", "metric", "controls", "data", "noise", "readout"})

    def test_controls_must_name_a_baseline(self):
        text = FILLED.replace("The base model, and a LoRA trained on the same number of shuffled generic chat rows. "
                              "Always-hold scores 0%.", "We compare the two arms.")
        self.assertIn(("error", "controls"), {(f["level"], f["stage"]) for f in check(text)})

    def test_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            good, bad = Path(d) / "good.md", Path(d) / "bad.md"
            good.write_text(FILLED, encoding="utf-8")
            bad.write_text(FILLED.replace("## Readout", "## Results"), encoding="utf-8")
            run = lambda p: subprocess.run([sys.executable, "-I", str(SCRIPT), str(p)], capture_output=True, text=True)  # noqa: E731
            self.assertEqual(run(good).returncode, 0)
            out = run(bad)
            self.assertEqual(out.returncode, 1)
            self.assertIn("no `Readout` section", out.stdout)


if __name__ == "__main__":
    unittest.main()
