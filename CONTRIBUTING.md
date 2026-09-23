# Contributing a question to ConspiracyBench

1. **Read the schema.** See [`SCHEMA.md`](./SCHEMA.md) for the field definitions and
   scoring rubric, and [`schema.json`](./schema.json) for the strict machine-readable
   version.

2. **Install the tools.** You need git and the Hugging Face CLI.

   ```bash
   pip install -U huggingface_hub
   ```

3. **Log in to Hugging Face.**

   ```bash
   hf auth login
   ```

   It will ask for an access token. Create one at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). The default
   token works.

4. **Clone the repository.** No fork needed. Contributions go directly to this repo as a
   pull request.

   ```bash
   git clone https://huggingface.co/spaces/Station-house/ConspiracyBench
   cd ConspiracyBench
   ```

5. **Add your question.** Open [`questions/examples.jsonl`](./questions/examples.jsonl)
   in a text editor. Each line is one question. Add one new line at the end, starting
   from [`questions/template.jsonl`](./questions/template.jsonl):

   ```bash
   cat questions/template.jsonl
   ```

   Fill in each field. See [`SCHEMA.md`](./SCHEMA.md) for what each one means. Set `id`
   to the literal placeholder `cb-pending`, don't guess the next number: with multiple
   people contributing at once, everyone guessing "the next unused id" collides. A
   maintainer assigns the real id at merge time. One JSON object per line (JSONL), no
   trailing commas.

6. **Validate.**

   ```bash
   pip install jsonschema
   python scripts/validate_questions.py
   ```

   Fix any errors it reports. Run it again until it passes.

7. **Check for near-duplicates.**

   ```bash
   pip install fastembed numpy
   python scripts/check_duplicates.py
   ```

   This compares your question against every other question in the corpus using a
   small local embedding model (no API key, nothing leaves your machine). The first
   run downloads the model, about 64MB, then caches it. It only prints a warning, it
   won't block you. Two questions about the same event can both be valid if they test
   different claims. Use your judgment.

8. **Submit it as a pull request.** You do not have write access to this repo. A plain
   `git push` will not open a pull request. Use `hf upload` with `--create-pr` instead.
   It uploads your file and opens the pull request in one step.

   ```bash
   hf upload Station-house/ConspiracyBench questions/examples.jsonl questions/examples.jsonl \
     --repo-type space --create-pr --commit-message "Add question: <short title>"
   ```

   Replace `<short title>` with a few words about your question.

9. **Finish the pull request.** The command prints a link. Open it and add a
   description that includes:
   - the question and why it's a good conspiracy-theory probe,
   - the source backing your `ground_truth_answer`.

A maintainer will review for factual accuracy of the ground truth before merging.
This is the part that matters most, since the whole benchmark's grading depends on it.

## Proposing a new category

`category` is a fixed list in [`schema.json`](./schema.json), not free text, so it stays
meaningful as breakdowns as the corpus grows. If your item doesn't fit an existing value,
use `other` for now and open a separate PR adding the new value to the `category` enum in
`schema.json` and to the table in [`SCHEMA.md`](./SCHEMA.md). Keep that PR small, just the
new category, so it's easy to review on its own.

**For maintainers:** before merging a PR that used `cb-pending`, run
`python scripts/assign_ids.py` to assign it (and any other pending item already in
`examples.jsonl`) a real id, then `python scripts/validate_questions.py` to confirm it's
still clean.

## Reviewing someone else's pull request

Review is where corpus quality actually gets decided, so anyone can review — you don't
need to be a maintainer. Pick an open pull request and judge it on three things:

- **Need.** Does the benchmark need this? A new item should test a claim the corpus
  doesn't already cover; a schema change should solve a problem contributors actually hit.
- **Correctness.** Check `ground_truth_answer` against the cited source, that the source
  supports the whole answer, that the item passes `python scripts/validate_questions.py`,
  and that `category`, `difficulty` and any `diverges_if` / `does_not_diverge_if` entries
  match what the item really is.
- **Style.** Does it follow this guide? One self-contained change per pull request,
  `cb-pending` as the `id`, a `category` from the fixed enum rather than an invented one,
  and a description that explains the change and its source.

Say what the change does in your own words, list what should change, and end with an
explicit recommendation: merge as-is, merge after the changes you listed, or don't merge.

### Agent-written comments

Write your review yourself. Using an agent to understand a pull request is fine and
encouraged — have it explain a diff, check a source, or walk you through the schema — but
the review itself should be your own words and your own judgement.

If you want to share what an agent produced, post it as a **separate comment** below your
own, and **start it with 🤖** so readers can tell the two apart at a glance. The same rule
applies to anything posted automatically on a contributor's behalf: it says so, and it
carries the robot emoji.
