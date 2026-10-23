# ConspiracyBench baseline: five ~4B thinking models (2026-10-23)

Qwen3.5 4B is the baseline; Gemma 4 E4B, Granite 4.2 3B, Nemotron 3 Nano 4B and K2 Horizon 3.7B
are comparable in size. Every answer was graded by pplx-decider v1.1 27B, the only judge.

**Interactive results:** <https://spacesheep.dev/@likelytobelaura/conspiracybench-baseline>.
It covers every metric with its interval, divergence turn by turn under pressure, consistency
across the five samples, breakdowns by category, difficulty, framing and region, the judge's
confidence, and a question explorer with transcripts for the 60 questions where the models
disagree most. Source in [`page/`](page/).

## Setup

| | |
|---|---|
| Corpus | `questions/corpus.jsonl` at `main` 67b62b375: 315 items, 254 multi-turn (727 pressure turns), 61 single-turn |
| Evals | Three, with no item asked as a single-turn question twice: **multi-turn** (254 items, the whole conversation in each of 5 epochs), **first question held out** (the same 254 items' opening question alone, 5 samples), **single-turn** (61 items, 5 samples). 6,480 model turns per model |
| Models | Thinking on for all (Gemma 4 forced on, K2 Horizon at `reasoning_effort` high, its default), each model's recommended sampling (`generation_config.json`, or the model card for Qwen3.5 4B and K2 Horizon, which have none), 8,192-token reply cap, 32k context, earlier answers sent back without their reasoning |
| Serving | vLLM 0.32.0 on one H100 per model (Modal), Inspect 0.3.283 solver, `inspect_eval/modal_solver.py` |
| Judge | pplx-decider v1.1 27B @ 14c9d901 on H100, endorsement rules (`rubric_v2`), the whole earlier conversation as context on later turns (`inspect_eval/decision_judges/`). Validated on the final labels (reference v1.2): 90.7% agreement, κ .755 |
| Invalid | Empty and cut-off replies are not graded and are left out of the rates (reported below) |
| Code | branch `worktree-cb-eval-baseline-prep`; solver at f73fced44 (K2 retries at 1b2fcca27), grading at 856d9cccb |

## Results

Rates with 95% Wilson intervals. Opening rates pool all 5 samples; concession and retreat are over
the epoch-1 conversations that open with hold. A conversation passes only if every turn holds.

| Metric | Qwen3.5 4B (baseline) | Gemma 4 E4B | Granite 4.2 3B | Nemotron 3 Nano 4B | K2 Horizon 3.7B |
|---|---|---|---|---|---|
| Multi-turn: opening divergence | 4.3% [3.3–5.6] | 3.9% [2.9–5.1] | 6.1% [4.9–7.6] | 14.3% [12.4–16.4] | 14.2% [12.3–16.4] |
| Multi-turn: concession (hold→diverge, epoch 1) | 4.1% [2.3–7.4] | 7.9% [5.1–12.1] | 2.8% [1.3–5.9] | 7.9% [4.9–12.4] | 5.1% [2.8–9.1] |
| Multi-turn: retreat (hold→abstain, epoch 1) | 0.0% | 0.0% | 0.0% | 0.5% | 0.0% |
| Multi-turn: conversation pass@5 | 96.7% [93.7–98.3] | 90.9% [86.7–93.9] | 94.5% [90.9–96.7] | 90.4% [85.9–93.6] | 90.9% [86.7–93.9] |
| Multi-turn: conversation worst-of-5 | 76.4% [70.7–81.3] | 70.0% [64.0–75.3] | 71.1% [65.3–76.4] | 43.0% [36.8–49.5] | 37.4% [31.6–43.7] |
| Multi-turn: conversation all-5-agree | 79.7% | 78.7% | 75.5% | 52.6% | 44.4% |
| First Q (held out): divergence | 5.4% [4.3–6.8] | 4.7% [3.6–6.0] | 5.6% [4.4–7.0] | 12.1% [10.4–14.1] | 13.8% [11.9–15.9] |
| First Q (held out): abstention | 0.4% | 2.0% | 5.5% | 5.1% | 9.0% |
| First Q (held out): pass@5 | 97.9% | 97.6% | 97.6% | 97.2% | 95.5% |
| First Q (held out): worst-of-5 | 90.5% [86.1–93.6] | 89.0% [84.5–92.3] | 76.8% [71.2–81.5] | 60.2% [54.0–66.0] | 54.5% [48.2–60.7] |
| Single-turn items: divergence | 2.4% [1.2–4.8] | 1.9% [0.8–4.3] | 7.7% [5.0–11.5] | 7.5% [5.0–11.1] | 7.4% [4.8–11.3] |
| Single-turn items: abstention | 3.3% | 12.8% | 14.4% | 3.9% | 15.2% |
| Single-turn items: pass@5 | 98.3% | 90.2% | 86.9% | 96.7% | 98.3% |
| Single-turn items: worst-of-5 | 86.7% [75.8–93.1] | 75.4% [63.3–84.5] | 68.9% [56.4–79.1] | 70.5% [58.1–80.4] | 40.7% [29.1–53.4] |
| Invalid turns (all evals) | 45 / 6,480 | 1 / 6,480 | 1 / 6,480 | 59 / 6,480 | 36 / 6,480 |

Clustered intervals (by `claim_id`), breakdowns by category, band, framing and region, and invalid
turns by reason are in each model's `report/pplx_<eval>.json`.

## Reading it

- **Qwen3.5 4B and Gemma 4 E4B lead on the opening question** (4–5% divergence) and are the most
  consistent across samples (all-5-agree ~80–93%). Qwen holds best under pressure: the lowest
  concession among the two (4.1% vs 7.9%) and the highest conversation worst-of-5 (76%).
- **Nemotron 3 Nano and K2 Horizon diverge about three times as often** on the opening question
  (12–14%) and are much less consistent: for under half the items do all 5 conversations hold on
  every turn.
- **pass@5 hides this.** At 90–98% it barely separates the models; worst-of-5 and all-5-agree do.
- **Abstention differs by model**: Qwen almost never declines (≤3%); Gemma, Granite and K2 decline
  12–15% of single-turn items. Nemotron's refusals often still state the facts and were graded hold.

## Caveats

- **One judge.** pplx agrees with the final human labels at κ .755 and errs towards over-calling
  diverge on borderline replies; several p ≈ 0.5–0.7 verdicts seen during grading are worth a
  human spot-check before these numbers are quoted.
- **K2 Horizon's non-English output.** On some of the 8 non-English items (Hindi, Arabic, Portuguese,
  French, Spanish) K2 produced garbled multi-script text, including leaked special tokens. Its
  rates include those items.
- **Truncation.** 8,192 tokens was enough for all but 0.02–0.9% of turns; those are invalid and left
  out of the rates, which may slightly favour models that think long on hard items (Qwen, Nemotron).
- **K2 Horizon's multi-turn eval** needed two `eval-retry` passes for one conversation (cb-0175,
  epochs 2 and 4): its template rejected a reply with no thinking field. All three logs are kept;
  the last one is complete and is what was graded.

## Files

This folder holds the write-up (`RESULTS.md`) and the results page source (`page/`). Every eval
output is in [`outputs/`](outputs/), one folder per model, and `outputs/smoke-2026-10-23/` holds the smoke tests.

Per model in `outputs/<model>/`: the Inspect logs per eval (`multiturn/`, `multiturn_first/`,
`single_turn/`; every message, reasoning, stop reason and model call), `inspect_<eval>.log`,
`vllm_server.log`, `manifest.json` (settings, sampling, commit, timings, exit statuses, retries),
`pip_freeze.txt`, `nvidia_smi.txt`, `units_sources.json` (the logs that were graded), `pplx/` (labels,
probabilities, run log) and `report/` (`inspect_eval/decision_judges/decision_report.py`). The turns as
graded (`units.jsonl`, 60–100 MB per model) aren't stored in the Space: `inspect_eval/export_units.py`
regenerates them from the logs. Everything is also on the Modal volume `cb-judges-results`
(`solver/<model>/full-2026-10-23/`, `eval/full-2026-10-23-<model>/`).

Compute: about $19 for the five solver runs and about $21 for grading (H100 on Modal).
