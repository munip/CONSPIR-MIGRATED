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

   *Shortcut:* `pipeline/auto/` can draft a claim in all three framings (each multi-turn) from any URLs or
   files with a local model, and walks you through verifying it; see
   [`pipeline/auto/GUIDE.md`](./pipeline/auto/GUIDE.md). You still review every claim, and steps 6-7 still apply.

5. **Add your question.** Create a new file in [`questions/pending/`](./questions/pending/),
   named after your submission, for example `questions/pending/wtc7-collapse.jsonl`. Do
   not add new questions to `questions/corpus.jsonl`; only the maintainer script adds them.
   (Adding pressure turns, with any checklist lines they need, or a `claim_id` to an item that is
   already in the corpus is the one exception; see [Adding to an item that is already in the corpus](#adding-to-an-item-that-is-already-in-the-corpus).)
   Start each line from [`questions/template.jsonl`](./questions/template.jsonl):

   ```bash
   cp questions/template.jsonl questions/pending/<short-slug>.jsonl
   ```

   Fill in each field. See [`SCHEMA.md`](./SCHEMA.md) for what each one means. Leave `id`
   as the literal placeholder `cb-pending`, don't guess the next number: with multiple
   people contributing at once, everyone guessing "the next unused id" collides. A
   maintainer assigns the real id after merging. One JSON object per line (JSONL), no
   trailing commas. Several questions in one file is fine; one file per pull request.

6. **Validate.**

   ```bash
   pip install jsonschema
   python scripts/validate_questions.py
   ```

   Fix any errors it reports. Run it again until it passes. Always run this before step 7.

7. **Check for near-duplicates.** Run this after `validate_questions.py` passes.

   ```bash
   pip install fastembed numpy
   python scripts/check_duplicates.py
   ```

   Only your pending items are checked. What it prints depends on your item's `category`:

   - **A defined category** (`space`, `vaccines`, `terrorism`, ...): every question
     already filed under that category, with your items marked `->`. Read the list and
     check whether your question is already covered.
   - **`other`**: pairs whose wording is similar (score 0.6 or higher), found with a
     small local embedding model.
   - **Every category**: pairs in *different* categories that score 0.8 or higher. These
     usually mean one of the two items is miscategorised.

   The embedding model runs locally (no API key, nothing leaves your machine). The first
   run downloads it, about 64MB, then caches it. Pass `--all` to print the whole corpus
   grouped by category.

   The script only reports, it won't block you. Two questions about the same event can
   both be valid if they test different claims. If yours is the same question reworded,
   drop it. If you are deliberately adding another framing of an existing claim, set
   your item's `claim_id` to the existing item's `claim_id` (see [`SCHEMA.md`](./SCHEMA.md)). If a cross-category pair shows your item is in the
   wrong category, fix its `category`.

8. **Write your PR description, and check it.** Every pull request needs a description that
   includes:
   - the question and why it's a good conspiracy-theory probe,
   - the source backing your `ground_truth_answer`.

   Write it to a file, then check it:

   ```bash
   cat > pr_description.md << 'EOF'
   Adds a question on <topic>: <what the question tests and why it's a good probe>

   Ground truth source: <url>
   EOF

   python scripts/check_pr.py pr_description.md
   ```

   Replace every `<...>` with your own words. Fix anything it flags and re-run until it
   passes. It's a minimum-bar check (non-empty, not placeholder text): it can't judge
   whether your description is good, only whether one exists.

9. **Submit it as a pull request.** You do not have write access to this repo. A plain
   `git push` will not open a pull request. Use `hf upload` with `--create-pr` instead.
   It uploads your file, opens the pull request and attaches your description in one step.

   ```bash
   hf upload Station-house/ConspiracyBench questions/pending/<short-slug>.jsonl questions/pending/<short-slug>.jsonl \
     --repo-type space --create-pr --commit-message "Add question: <short title>" \
     --commit-description "$(cat pr_description.md)"
   ```

   Replace `<short-slug>` with your file name and `<short title>` with a few words about
   your question. Because you only upload your own file, your pull request cannot
   conflict with anyone else's.

A maintainer will review for factual accuracy of the ground truth before merging.
This is the part that matters most, since the whole benchmark's grading depends on it.

## Finding and claiming work with the `stationhouse` CLI

The [`stationhouse` CLI](https://github.com/likelytobelaura/stationhouse-cli) (alias `sth`)
is the quickest way to navigate Station House from the terminal: see which tasks are open,
who is already on one, and put yourself on it. It signs in with your Station House account
(the same one you use on the website), and many people can share a task, so claiming never
locks it.

```bash
pip install git+https://github.com/likelytobelaura/stationhouse-cli
stationhouse login              # your Station House email and password, entered locally
stationhouse unclaimed          # open tasks nobody is on
stationhouse who <task-id>      # who is on a task, and whether they're active or stale
stationhouse claim <task-id>    # put yourself on it
stationhouse mine               # your own claims
stationhouse release <task-id>  # take yourself off it
```

Add `--json` to any command for output a script or agent can parse. `stationhouse
skills-install` installs a skill that lets a Claude Code agent run these commands for you;
it will ask you to run `login` yourself and only claims a task once you agree. Check `who`
before claiming so you can coordinate with anyone already on the task. The CLI only covers
finding and claiming work; submitting is still steps 1-9 above.

## Designing an experiment

Most experiments that go wrong were set up wrong, not run wrong: a fine-tune that beat nothing
because nobody ran the base model, two arms that differed in more than the thing being tested, a
difference smaller than the run-to-run noise. Settle the design before you run anything, write it
into `EXPERIMENT.md`, and have someone check it. This applies to analyses in
[`analysis/`](./analysis/) too whenever they compare groups or claim an effect.

Start from [`experiment_requests/TEMPLATE.md`](./experiment_requests/TEMPLATE.md). It has six stages,
in this order:

1. **Claim.** One falsifiable sentence: "If X, then <metric> on <which items> moves by at least Δ
   compared with Y." If you can't write it, the experiment isn't ready.
2. **Unit and metric.** Say what one data point is (an item, a claim, a turn, a judge verdict),
   how many there are, and how the metric is aggregated. Name the judge and rubric that grade it.
   Check what the metric gives a constant answer: an always-hold model can look good on some
   metrics.
3. **Controls.** Every experiment needs at least one trivial baseline: the base or unmodified model,
   a constant policy (always hold, always diverge), or the majority label. It also needs one matched
   control that differs from the treatment in exactly one thing.
4. **Data.** Give the corpus commit, the SHA-256 of any fixture, and the reference labels you use.
   List every filter, with the row count after it. Keep the evaluation set apart, and split by
   `claim_id`, never by item.
5. **Noise.** Give the samples per item and the seeds, and say how you'll measure the noise:
   rerun, or bootstrap over claims. Name the smallest difference you'd treat as real. Run a small
   pilot first to check the plumbing. Pilot numbers don't count.
6. **Readout.** Before the run, write the numbers you'll report, the pass rule, and the result that
   would refute the claim. Null and negative results get a `RESULTS.md` too.

What you changed after the run goes under `## Deviations`; leave the six stages as they were.

### What goes wrong on ConspiracyBench

- **The judge is part of the measurement.** Grade with the default judge and rubric
  (pplx-decider, rubric v2 with context), validated against the reference labels in
  [`completed_experiments/judge_reliability/`](./completed_experiments/judge_reliability/). If your
  change alters how answers read, for example a fine-tune that writes longer or hedgier answers,
  hand-check 20 to 30 verdicts against the rubric before trusting the judge on them.
- **Arms leak through the text.** In [Experiment 005](./completed_experiments/judge_injection/experiments/005-injection-resistance/),
  the length padding in the v1 fixture wrote to the grader about the dataset, so "clean" and
  "attack" answers differed in more than the injection. Compare your arms side by side: length,
  format, quoted text, and anything that talks to the grader.
- **Framings of one claim aren't independent.** Items that share a `claim_id` test the same claim.
  Split, bootstrap and count by claim. If training or prompt-writing data mentions corpus claims,
  say so: that's contamination, not a result.
- **Small differences are often noise.** With a few hundred items and k=5 samples, a gap of a few
  points can come from sampling alone. Rerun or bootstrap before you call it.
- **Filters change the population.** For each filter (length, language, slurs, dedup), record the
  count before and after, and read 20 random dropped rows and 20 random kept ones.
- **Categories are uneven.** Per-category numbers rest on very different sample sizes (see
  [`analysis/category-sensitivity/`](./analysis/category-sensitivity/)). Report the count next to
  every rate.

### Tools

```bash
pip install git+https://github.com/likelytobelaura/stationhouse-cli
stationhouse design new experiment_requests/<task>/<short-slug>/EXPERIMENT.md   # the template
stationhouse design check experiment_requests/<task>/<short-slug>/EXPERIMENT.md # missing pieces (no model)
stationhouse design review experiment_requests/<task>/<short-slug>/EXPERIMENT.md  # a critique from the Station House assistant
```

`check` and `scripts/check_design.py` apply the same rules, and passing one is required before a
request is merged. They only find missing pieces, not bad ideas. For the thinking, ask a reviewer on the
Discussions, use `design review`, or use a coding agent: Claude Code, Codex and Cursor read
[`AGENTS.md`](./AGENTS.md) in this repo. The agent walks you through the stages one at a time,
asks before it fills anything in, and red-teams the design before you run it.

## Contributing an experiment

Experiments go in [`experiment_requests/`](./experiment_requests/), whether you will run them
yourself or need someone else's model or compute. If you don't have the access an experiment
needs, you don't need to run it yourself: write it up, and a maintainer can run it for you.

1. **Create a folder for it.** `experiment_requests/<task>/<short-slug>/`, named after the
   experiment, for example `experiment_requests/judge_injection/injection-resistance/`.
   Everything for the experiment goes inside it. Experiments can be grouped in sub-categories
   by the task they belong to: use the task's folder if one exists here or in
   [`completed_experiments/`](./completed_experiments/) (for example `judge_reliability/`,
   `judge_injection/`, `stage2_pilot/`), create one for a new task, or leave out `<task>/`
   if the experiment doesn't belong to a task.
2. **Write `EXPERIMENT.md`** from [`experiment_requests/TEMPLATE.md`](./experiment_requests/TEMPLATE.md),
   following [Designing an experiment](#designing-an-experiment): the claim, unit and metric,
   controls, data, noise and readout, then the models, datasets, sample sizes, seeds, hardware and
   anything else needed to reproduce the run.
3. **Check the design. This is required.** A request isn't merged until this passes:

   ```bash
   python scripts/check_design.py experiment_requests/<task>/<short-slug>/EXPERIMENT.md
   ```

   It fails if any of the six stages is missing or empty, or if Controls names no baseline. It
   prints warnings for gaps a reviewer will ask about. Fix every error, and keep a warning only if
   `EXPERIMENT.md` says why. Paste its output into your PR description. Reviewers run the same
   check on your PR. `stationhouse design check` applies the same rules.
4. **Add the code.** The code, config or scripts needed to run it, plus the exact commands.
   Reference large inputs that are already in the repo by path rather than copying them.
5. **Open a pull request** with the folder. Write and check your description as in step 8
   above (include the design check output), then upload the whole folder:

   ```bash
   hf upload Station-house/ConspiracyBench experiment_requests/<task>/<short-slug> experiment_requests/<task>/<short-slug> \
     --repo-type space --create-pr --commit-message "Experiment request: <short title>" \
     --commit-description "$(cat pr_description.md)"
   ```

6. **If you need compute, tag @stationhouse** in the pull request description and say what
   you need.

A maintainer will review the experiment and, if it's a good fit, run it or give you what you
need to run it. Once it has run and the results are merged, the folder moves to the same
task folder in [`completed_experiments/`](./completed_experiments/), with a `RESULTS.md` next
to the original `EXPERIMENT.md`. Results go in a pull request against the experiment's folder
in `experiment_requests/`; a maintainer does the move.

## Adding to an item that is already in the corpus

Pressure turns and extra framings can be added to a claim after its first item has merged (see
[`SCHEMA.md`](./SCHEMA.md) and the "Expand existing questions" task). New framings are new
questions: they go in a new file in `questions/pending/`, with the claim's `claim_id`. Adding
`pressure_turns`, or the shared `claim_id`, to an item that already has a real id is done on that
item's line in `questions/corpus.jsonl`:

- Change only that line, and only the fields you're adding: `pressure_turns`, the `claim_id`, and
  any `diverges_if` / `does_not_diverge_if` lines the new turns need. Keep it on one line,
  serialised the way `scripts/assign_ids.py` writes it (`json.dumps` with default settings), so no
  other line changes.
- Don't put an edited copy of the item in `questions/pending/`. `assign_ids.py` would give it a
  new id, and the corpus would end up with the item twice.
- If the item is still in someone else's open pull request, don't edit it. Comment on that pull
  request, or wait until it merges.
- Start from a fresh `git pull`, and upload `corpus.jsonl` together with any new pending file in
  one pull request:

  ```bash
  hf upload Station-house/ConspiracyBench questions questions \
    --repo-type space --create-pr \
    --include "corpus.jsonl" --include "pending/<claim-slug>-framings.jsonl" \
    --commit-message "Expand <claim-slug>: add framings and pressure turns"
  ```

  This uploads your copy of `corpus.jsonl` whole, so a stale copy would undo other people's
  merged changes.
- Git treats edits to neighbouring lines as a conflict. If another open pull request edits the
  line just above or below yours, whichever merges second will need updating.

## Proposing a new category

`category` is a fixed list in [`schema.json`](./schema.json), not free text, so it stays
meaningful as breakdowns as the corpus grows. If your item doesn't fit an existing value,
use `other` for now and open a separate PR adding the new value to the `category` enum in
`schema.json` and to the table in [`SCHEMA.md`](./SCHEMA.md). Keep that PR small, just the
new category, so it's easy to review on its own.

**For maintainers:** don't merge or run an experiment request until
`python scripts/check_design.py` passes on its `EXPERIMENT.md`. The review-pr skill's `prep_pr.sh` runs
it for you. Then merge a PR that added a file to `questions/pending/` as it is, then run
`python scripts/assign_ids.py` on `main`. It assigns real ids to every pending item, appends
them to `questions/corpus.jsonl`, and deletes the pending files. Then run
`python scripts/validate_questions.py` to confirm the corpus is still clean, and commit the
result to `main`. Don't run it on a PR branch before merging: every branch starts from the
same corpus, so two PRs processed that way both get the same next id and conflict on
`corpus.jsonl`. You can run it after each merge or once after merging several PRs; pending
files are processed in submission order either way.

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

## For AI coding agents

If an AI coding agent wrote or drafted your pull request, say so in the description. A last
line starting with 🤖 is enough, for example:

> 🤖 Drafted with an AI coding agent. I checked the sources and ran `validate_questions.py` before opening this PR.

You are still the author. Check every source yourself and stand behind the change. Review
comments follow the rule above: your review in your own words, the agent's output in a
separate comment that starts with 🤖.

If you are an agent working in this repository:

- Follow steps 1-9 above, or the task page's instructions when you're editing items already in the
  corpus. Leave `id` as `cb-pending`, and never run `scripts/assign_ids.py` on a pull request branch.
- Don't open pull requests, post comments or claim tasks for a contributor without their
  go-ahead, and mark anything you post with 🤖.
- For experiments and analyses, follow [`AGENTS.md`](./AGENTS.md) and [Designing an experiment](#designing-an-experiment).
  Help the contributor make the design choices; don't make them yourself, and don't run anything before
  the Readout is written.
- Don't ask for or handle a contributor's passwords or tokens. `hf auth login` and
  `stationhouse login` are for the contributor to run.
