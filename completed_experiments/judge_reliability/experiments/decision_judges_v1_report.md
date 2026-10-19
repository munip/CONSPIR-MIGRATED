# Decision judges v1: Clef, Kev-4B, Laya and pplx-decider on the validity set

The GPU run for #203 and #208: the four decision-model candidates on all 150 real-transcript units
(244 calls each), under one fixed setup, scored against the 150-unit human reference (#364) with the
pre-registered `score_validity.py`. Run 2026-10-16 03:47–03:54 UTC on Modal. Kit:
`completed_experiments/judge_reliability/decision_judges_v1/` (protocol, locked environment, Modal app).

## Setup (the same for every judge)

- One Modal container per judge, all of one spec: `H100!` (NVIDIA H100 80GB HBM3, driver 610.57.04,
  never upgraded to an H200), 8 CPUs, 128 GB RAM. Each container ran one judge in a fresh process.
- One locked environment: Python 3.12, torch 2.14.0+cu130, transformers 5.17.0, flash-linear-attention
  0.5.2. Kev declares `torch<2.9`; it passed the smoke test under the override, so all four ran in the
  same environment. `pip_freeze.txt` and `nvidia_smi_*.txt` are saved per judge.
- The same 244 requests in the same order, batch size 1, no truncation. All models were downloaded to a
  volume at their pinned revisions before any timed run. Every judge did 3 untimed warm-up calls.
- Smoke test first (3 calls per judge, `*_validity_v1_smoke/`), then the full run. **All four: 244/244
  calls, 0 errors, 0 invalid outputs.** No unit exceeded any judge's input limit.

## Validity (reference = #364, all 150 units; `decision_judges_v1_validity.txt` has every split)

| Judge / variant | n | κ | Diverge recall [95% CI] | Diverge precision | Diverge rate judge / human | Hold recall | Abstain recall |
|---|---|---|---|---|---|---|---|
| clef/no_context | 150 | 0.421 | 33/34 [0.85, 0.99] | 0.45 | 0.49 / 0.23 | 69/111 | 1/5 |
| clef/preceding_exchange | 94 | 0.406 | 24/25 [0.80, 0.99] | 0.48 | 0.53 / 0.27 | 39/65 | 0/4 |
| kev/no_context | 150 | 0.398 | 19/34 [0.39, 0.71] | 0.66 | 0.19 / 0.23 | 77/111 | 5/5 |
| kev/preceding_exchange | 94 | 0.412 | 15/25 [0.41, 0.77] | 0.54 | 0.30 / 0.27 | 48/65 | 2/4 |
| laya/no_context | 150 | -0.064 | 8/34 [0.12, 0.40] | 0.19 | 0.29 / 0.23 | 77/111 | 0/5 |
| laya/preceding_exchange | 94 | 0.048 | 1/25 [0.01, 0.20] | 1.00 | 0.01 / 0.27 | 65/65 | 0/4 |
| pplx/no_context | 150 | 0.580 | 31/34 [0.77, 0.97] | 0.62 | 0.33 / 0.23 | 85/111 | 3/5 |
| pplx/preceding_exchange | 94 | 0.598 | 23/25 [0.75, 0.98] | 0.62 | 0.39 / 0.27 | 53/65 | 0/4 |

Human–human κ on the units with two human sheets (u076–u150, before adjudication): 0.497.

What the numbers say, against the #203 protocol:

- **Diverge recall (criterion 1):** Clef 33/34 and pplx-decider 31/34; their intervals overlap. Kev 19/34
  and Laya 8/34 are well below.
- **κ (criterion 2):** pplx-decider is highest (0.58, 0.60 with the preceding exchange), above the
  human–human 0.50 on the second half. Clef and Kev are about 0.40–0.42; Laya is at chance.
- **Over-calling (the precision gate proposed in #323: diverge precision ≥ 0.80):** no judge passes.
  Clef labels 49% of replies diverge where humans label 23%, so it would roughly double a published
  divergence rate. pplx-decider over-calls less (33% vs 23%). These precisions are raw; the reweighted
  precision needs the pre-registered real-run class mix, which was not passed (`--class-mix`), so it is
  not reported here.
- **Abstain (criterion 3):** only 5 abstain units, so every interval is wide. Kev got 5/5 without
  context.
- **Context:** adding the preceding exchange changes little for Clef, Kev and pplx-decider. It makes
  Laya label almost everything hold (65/65 hold, 1/25 diverge).
- **Invalid rate (gate 2):** 0 for all four.

## Speed and cost (`results/decision_judges_v1_speed/`)

Latency over timed successful calls; cost at $5.35 per container-hour (Modal list prices: H100 $3.95/h +
8 cores + 128 GiB).

| Judge | OK / calls | Load s | Mean s | p50 s | p90 s | p99 s | Calls/s | Tokens/s | Peak GB | Mean W | Wh | GPU min | Cost $ | $ / 1k calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clef | 244 / 244 | 36 | 0.404 | 0.322 | 0.556 | 0.979 | 2.48 | 5029 | 56.0 | 315 | 17.2 | 3.2 | 0.29 | 0.600 |
| kev | 244 / 244 | 38 | 0.150 | 0.104 | 0.146 | 0.834 | 6.65 | 11894 | 9.4 | 182 | 6.4 | 2.0 | 0.18 | 0.223 |
| laya | 244 / 244 | 17 | 0.167 | 0.158 | 0.212 | 0.253 | 5.98 | 11042 | 1.8 | 113 | 2.3 | 1.0 | 0.09 | 0.249 |
| pplx | 244 / 244 | 75 | 0.311 | 0.226 | 0.494 | 0.631 | 3.21 | – | 53.2 | 301 | 17.5 | 3.4 | 0.30 | 0.463 |

Clef's 0.40 s mean matches #126's 0.43 s per call. pplx-decider doesn't report its input size, so it
has no tokens/s. GPU time includes model load and warm-up. The whole run, including downloads and the
smoke test, cost a few dollars.

## Caveat: how the reference was made, by slice (`results/decision_judges_v1_slices.md`)

Every reference label is a human's (#364). On u001–u020 and u076–u150 there were two human labellers,
plus adjudication. On u021–u075 the second labeller was Qwen3-32B, and every disagreement went to the
human (powercoding1), so that slice rests on one person. Clef, Kev and pplx-decider are Qwen-based;
Laya is not.

| Judge (no_context) | κ, two-human slice (n=95) | κ, one-human slice (n=55) |
|---|---|---|
| clef | 0.317 | 0.600 |
| kev | 0.289 | 0.538 |
| laya | -0.254 | 0.091 |
| pplx | 0.399 | 0.861 |

All three Qwen-based judges agree much more on the one-human slice. That fits the concern that the
Qwen second label could favour Qwen judges: units where Qwen3-32B disagreed with powercoding1 still have
the human's label, but the slice was only checked against a Qwen model. The slices also differ in units
and in which humans labelled them, so this is a flag, not a finding. The two-human numbers are the
conservative ones.

## Other limits

- The four containers ran in parallel on different physical H100 hosts of one spec, not on one machine.
- Source-model splits (#203 gate 1) need `key.csv`, which powercoding1 holds, so they are not run here.
- No cascade or jury: those need the chat-judge predictions too (`score_validity.py --cascade/--jury`).
- 5 abstain units cannot rank judges on abstain.

## Files

- `results/<judge>_validity_v1/`: `results.jsonl` (every call: label, confidence, probabilities, input
  tokens, latency, session), `sessions.jsonl` (load time, warm-up, peak memory, runtime versions),
  `manifest.json`, `gpu_telemetry.csv` (1 s nvidia-smi samples), `nvidia_smi_*.txt`, `pip_freeze.txt`,
  `run.log`, `run_info.jsonl`. `<judge>_validity_v1_smoke/` is the same for the smoke test.
- `results/decision_judges_v1_validity.{txt,json}`: full `score_validity.py` output.
- `results/decision_judges_v1_validity_summary.md`, `results/decision_judges_v1_slices.{md,json}`,
  `results/decision_judges_v1_speed/speed_summary.{md,json}`.
- `results/decision_judges_v1_logs/`: the Modal download, smoke and full-run logs.
