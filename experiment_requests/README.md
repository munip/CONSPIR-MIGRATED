# Experiment requests

Experiments that are proposed but not yet run. Each one is a pull request that adds its own
folder here, typically because it needs a model or compute the contributor doesn't have.
See [Contributing an experiment](../CONTRIBUTING.md#contributing-an-experiment) for how to
submit one.

An experiment that belongs to a task goes in a sub-folder named after that task, for example
`experiment_requests/judge_injection/<short-slug>/` for a new injection study. Use an
existing task folder from here or from [`completed_experiments/`](../completed_experiments/)
when one fits. Otherwise create a new one, or put the experiment directly in
`experiment_requests/`.

```
experiment_requests/[<task>/]<short-slug>/
  EXPERIMENT.md   research question, parameters, how to run it, what counts as a result
  ...             code, configs and any small input files the run needs
```

Once an experiment has been run and its results are merged, a maintainer moves its folder
to the same task folder in [`completed_experiments/`](../completed_experiments/).
