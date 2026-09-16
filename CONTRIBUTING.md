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

   Fill in each field. See [`SCHEMA.md`](./SCHEMA.md) for what each one means. Give your
   item a fresh `id`, the next unused `cb-XXXX`. One JSON object per line (JSONL), no
   trailing commas.

6. **Validate.**

   ```bash
   pip install jsonschema
   python scripts/validate_questions.py
   ```

   Fix any errors it reports. Run it again until it passes.

7. **Submit it as a pull request.** You do not have write access to this repo. A plain
   `git push` will not open a pull request. Use `hf upload` with `--create-pr` instead.
   It uploads your file and opens the pull request in one step.

   ```bash
   hf upload Station-house/ConspiracyBench questions/examples.jsonl questions/examples.jsonl \
     --repo-type space --create-pr --commit-message "Add question: <short title>"
   ```

   Replace `<short title>` with a few words about your question.

8. **Finish the pull request.** The command prints a link. Open it and add a
   description that includes:
   - the question and why it's a good conspiracy-theory probe,
   - the source backing your `ground_truth_answer`.

A maintainer will review for factual accuracy of the ground truth before merging.
This is the part that matters most, since the whole benchmark's grading depends on it.
