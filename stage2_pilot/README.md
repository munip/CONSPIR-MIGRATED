# Stage 2 evidence-ledger pilot

This pilot deterministically renders manually authored evidence ledgers
into ConspiracyBench's existing question schema. It includes six
draft examples with supporting source notes: three single-turn and three multi-turn.

The renderer does not research facts, generate new claims, verify source
contents, fetch live PRs, or automatically approve questions.

## Contents

- `renderer.py`: validation, deterministic rendering, and export safeguards.
- `fixtures/synthetic_ledger.json`: fictional software-test input.
- `ledgers/six_question_drafts.json`: six real-topic drafts.
- `tests/test_renderer.py`: offline regression and draft-preview tests.
- `EXPERIMENT.md`: pilot results and limitations.

## Run the tests

From the repository root:

```powershell
python -B -m unittest discover -s stage2_pilot/tests -p "test_*.py" -v
```

The six-draft integration test temporarily overrides review status in
memory to check schema compatibility. It does not change the saved
ledger or assert independent human verification.

## Review and export requirements

Real-question exports require an inventory of the merged corpus and
active question PRs. Exact question and claim-ID checks are built in;
semantic overlaps require separate review.

All six examples remain drafts. Their source mappings are preliminary
notes for maintainer review before benchmark inclusion.

The renderer refuses to export synthetic fixtures by default, write
directly to `questions/`, or overwrite existing candidate files.

The synthetic example uses an intentionally invalid source URL and
must never be submitted as a benchmark question.
