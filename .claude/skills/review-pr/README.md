# review-pr: a Claude Code skill for the daily "Review a PR" task

This skill does the legwork of a ConspiracyBench review and gives you notes. You still write the review yourself, as the task requires.

What it does:
- Fetches the PR into a throwaway git worktree. Your own clone and your unsubmitted pending files are untouched.
- Runs main's copies of `validate_questions.py` and `check_duplicates.py` on just this PR's items. If the PR changes code, it warns you and shows the diff. The PR's own scripts run only if you pass `--run-pr-scripts` after reading them.
- Checks every clause of the ground truth against the cited sources, with quotes.
- Checks framing, difficulty, region, pressure turns and the diverges_if lists against `SCHEMA.md`.

You get bullet notes under Need, Correctness and Style, plus the recommendation options.

It never writes your review and never posts to Hugging Face. If you want agent output on the PR, it can draft a separate follow-up comment starting with 🤖, per the review guidelines.

## Use it

1. Clone the repo, if you haven't already:

   ```bash
   git clone https://huggingface.co/spaces/Station-house/ConspiracyBench
   ```

2. Open Claude Code inside the clone. Skills in `.claude/skills/` load automatically.

3. Ask it something like:
   - "find me a ConspiracyBench PR to review"
   - "review PR 191"
   - or paste the discussion link.

The scripts also work without Claude:

```bash
python .claude/skills/review-pr/scripts/list_open_prs.py --me <your HF username>
bash .claude/skills/review-pr/scripts/prep_pr.sh <PR number>
bash .claude/skills/review-pr/scripts/prep_pr.sh --run-pr-scripts <PR number>   # only after reading the PR's code
```

**Needs:** git, curl, python3 and bash. On Windows, use WSL or Git Bash.

The first run creates a small venv in `~/.cache/conspiracybench-review/` and downloads the ~64MB embedding model used by `check_duplicates.py`.
