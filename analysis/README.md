# Analysis results

Results from analysing benchmark outputs, starting with the Phase 2 baseline run in
[`completed_experiments/evals/full-2026-10-23/`](../completed_experiments/evals/full-2026-10-23/).
An analysis here reads existing outputs; it doesn't train or run models. Experiments that do go in
[`experiment_requests/`](../experiment_requests/).

Each analysis gets its own folder:

```
analysis/<short-slug>/
  RESULTS.md   the question, what you found, the numbers and example transcripts behind them,
               and which outputs you used
  ...          the code or notebook that produced the numbers, and any figures
```

## How to add one

1. Comment on the Space's [Discussions](https://huggingface.co/spaces/Station-house/ConspiracyBench/discussions)
   with what you want to look at.
2. Draft the analysis as a pull request that adds `analysis/<short-slug>/`:

   ```bash
   hf upload Station-house/ConspiracyBench analysis/<short-slug> analysis/<short-slug> \
     --repo-type space --create-pr --commit-message "Analysis: <short title>"
   ```

3. Link the draft PR from your discussion, then publish the PR when it's ready for review.

If the analysis compares groups or claims an effect, settle its design first: what you're comparing
against, how many items each number rests on, and what would count as noise. See
[Designing an experiment](../CONTRIBUTING.md#designing-an-experiment).

Someone else should be able to rerun your code and get the same numbers. Large outputs that are
already in the repo should be referenced by path, not copied.

## Analyses

| Folder | What it covers |
|---|---|
| [`category-sensitivity/`](./category-sensitivity/) | How baseline results vary by question category (#432) |
