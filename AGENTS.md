# Notes for AI coding agents

These notes are for any coding agent working in this repository (Codex, Cursor, Claude Code and others).
The general rules are in [CONTRIBUTING.md, "For AI coding agents"](./CONTRIBUTING.md#for-ai-coding-agents):
say in the description when an agent drafted a pull request, mark anything you post with 🤖, never
handle the contributor's passwords or tokens, and don't open PRs, post or claim anything without
their go-ahead. To review a PR, use [`.claude/skills/review-pr/SKILL.md`](./.claude/skills/review-pr/SKILL.md).

## Helping someone design an experiment or analysis

When a contributor asks how to set up an experiment, what a good control would be, how to clean
their data, or whether a comparison is fair, act as their design partner. Don't design it for them.
The method is in [CONTRIBUTING.md, "Designing an experiment"](./CONTRIBUTING.md#designing-an-experiment),
and the file is [`experiment_requests/TEMPLATE.md`](./experiment_requests/TEMPLATE.md).

1. **Read both first**, including "What goes wrong on ConspiracyBench". Look in
   `completed_experiments/` and `analysis/` for an earlier study of the same question. If one
   exists, say so and reuse its fixtures and judges.
2. **Start the file** at `experiment_requests/<task>/<short-slug>/EXPERIMENT.md` (copy the template,
   or `stationhouse design new <that path>`). For an analysis, put the same six sections at the top
   of `analysis/<short-slug>/RESULTS.md`.
3. **Go one stage at a time:** Claim, Unit and metric, Controls, Data, Noise, Readout. Ask about
   each stage and write down what they decide. When they're stuck, give them two or three concrete
   options with the trade-offs, and let them choose. Don't fill a stage with your own guess and
   move on.
4. **Push on the usual weak spots.**
   - Is there a trivial baseline: the base model, always-hold or always-diverge, the majority label?
   - Is there a matched control that differs in exactly one thing?
   - Is the split by `claim_id`?
   - Is the judge trustworthy on these answers?
   - Is the expected gap bigger than the noise from k samples and reruns?
5. **Check it** with `python scripts/check_design.py <file>`. `stationhouse design check <file> --json`
   applies the same rules. A request isn't merged until the check passes, so don't help open the PR
   before it does. Fix the errors with the contributor, and talk through each warning.
6. **Red-team it before anything runs.**
   - What's the dumbest method that would score well?
   - What else would explain a positive result?
   - Could evaluation items have leaked into what was trained, tuned or written?
   - Does the Readout say which result would mean "no"?
7. **Run a small pilot first, and only with their go-ahead.** Don't change Claim, Controls or
   Readout after results exist. Record changes under `## Deviations`.

Reviewing someone else's design: give notes per stage, with what's missing, the strongest
alternative explanation, and one concrete fix each. The review comment is the reviewer's own
words. Agent output goes in a separate comment that starts with 🤖.
