# Contributing a question to ConspiracyBench

1. **Read the schema.** See [`SCHEMA.md`](./SCHEMA.md) for the field definitions and
   scoring rubric, and [`schema.json`](./schema.json) for the strict machine-readable
   version.

2. **Fork this Space.** On the ConspiracyBench page, use the "⋮" menu → *Duplicate this
   Space*, or clone it directly:

   ```bash
   git clone https://huggingface.co/spaces/Station-house/ConspiracyBench
   cd ConspiracyBench
   ```

3. **Copy the template** and fill in one line per question in
   [`questions/template.jsonl`](./questions/template.jsonl):

   ```bash
   cp questions/template.jsonl /tmp/my-question.jsonl
   # edit /tmp/my-question.jsonl, then append it:
   cat /tmp/my-question.jsonl >> questions/examples.jsonl
   ```

   Give your item a fresh `id` (next unused `cb-XXXX`). One JSON object per line
   (JSONL), no trailing commas.

4. **Validate:**

   ```bash
   pip install jsonschema
   python scripts/validate_questions.py
   ```

5. **Open a pull request** against `Station-house/ConspiracyBench` with your change.
   Include in the PR description:
   - the question and why it's a good conspiracy-theory probe,
   - the source backing your `ground_truth_answer`.

A maintainer will review for factual accuracy of the ground truth before merging —
this is the part that matters most, since the whole benchmark's grading depends on it.
