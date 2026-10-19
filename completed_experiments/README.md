# Completed experiments

Experiments that have been run, with their results. Each one starts as an
[experiment request](../experiment_requests/) and is moved here by a maintainer once the run
is done and the results are merged.

Experiments are grouped in sub-folders by the task they belong to. A request that was filed
under a task folder moves to the folder with the same name here:

| Task folder | What it covers |
|---|---|
| [`judge_reliability/`](./judge_reliability/) | Choosing the judge: judge baselines, Clef and decision-model runs, the validity set v1 and its labels, chat-judge comparisons (#50, #113, #122, #126, #203, #208) |
| [`judge_injection/`](./judge_injection/) | Prompt-injection fixtures and detector studies (PromptShield, Prompt Guard 2, Kev/Laya), in `experiments/` and `reports/` (#111, #173) |
| [`stage2_pilot/`](./stage2_pilot/) | Stage 2 evidence-ledger pilot: rendering hand-written evidence ledgers into questions |

```
completed_experiments/<task>/<short-slug>/
  EXPERIMENT.md   the original request, unchanged
  RESULTS.md      what was run (models, versions, hardware, date), the numbers, and what they mean
  ...             code, configs and the raw outputs or a link to them
```

The three folders above were moved here from `eval/judge_reliability/`, `eval/judge_injection/`
and `stage2_pilot/`, with their own layouts kept as they were. Recorded results (JSON, CSV,
HTML, logs and source archives) were not edited, so paths written inside them still use the
old locations.
