# Judge reliability rerun on the final labels (reference v1.2)

Reruns the four decision judges from the judge reliability study (pplx-decider, Clef, Kev-4B,
Laya) on all 150 validity-v1 replies and scores them against
[`reference_v1.2.csv`](../validity_v1/labels/reference_v1.2.csv), the final labels (PR #424),
treated as ground truth. Qwen chat judges are excluded, as for the rest of the study.

## Setup

The deployed setup, as requested on 2026-10-22:

| Setting | Value |
|---|---|
| Rules | Endorsement rules (`--criteria rubric_v2`, `eval/judge_prompt.py:RUBRIC_V2_*`) |
| Context | Later turns (94 of 150 units) get the preceding exchange (`preceding_exchange`); opening turns run without context. Each later turn is also run without context for comparison, so 244 calls per judge |
| Checkpoints | As in [`decision_judges_v1/protocol.json`](../decision_judges_v1/protocol.json) |
| Hardware | One H100 80GB per judge on Modal (`decision_judges_v1/modal_app.py::rerun --tag rerun_v1.2_2026-10-22 --criteria rubric_v2`), four in parallel |
| Code | Commit `7094ee6e` (main at `4cefe5dc` plus PR #409's rubric options, plus a `--tag` option in the Modal app) |

## Outputs

- `../results/rerun_v1.2_2026-10-22/<judge>_validity_v1_rubric_v2/`: results, manifest, run log,
  sessions, GPU telemetry, pip freeze and `nvidia-smi` per judge. The `_smoke` folders are the 3-call
  smoke runs that preceded them.
- `score.py` → `scores.json`: every metric in [RESULTS.md](RESULTS.md), including bootstrap 95%
  intervals (2,000 resamples of units, seed 0).
