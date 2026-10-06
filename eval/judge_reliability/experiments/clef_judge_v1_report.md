# Clef judge experiment: report

**Question (PR #126):** can Cloudflare Clef serve as a stronger judge for ConspiracyBench than the local candidates tested in #122, under the same judge-reliability protocol?

**Status after Phases A, B and C (context panel): Clef is competitive on everything measured so far. No judge recommendation yet, because the first criterion in the decision rule, real-transcript agreement, has not been run.** Its inputs are not in the repository (see Phase C).

Model: `Cloudflare/clef` at revision `2f3de3dd85f379784083b0814d997ab627200f0c`, bfloat16, no quantization, on one NVIDIA A100 80GB (Runpod, $1.39/hr). Total GPU spend for all phases: about $1.

## Results

| Phase | What ran | Clef result | Detail |
|---|---|---|---|
| A. Preflight | 6 fixed calls (3 clean, 3 injected) | 6/6 valid and correct, 0 retries | `clef_judge_v1_preflight_report.md` |
| B. Standardized suite | The #122 84-call plan (same task-plan hash) | **60/60** core, 0/10 injection target-follows, 0 invalid, 12/12 repeat stability | `clef_judge_v1_phase_b_report.md` |
| C. Near-context stress | 9 calls at up to about 15,200 tokens (93% of the limit) | **9/9** | `clef_judge_v1_phase_c_report.md` |
| C. Real-transcript agreement | 20 responses from six real conversations | **not run: inputs missing** | `clef_judge_v1_phase_c_report.md` |

### Against the #122 candidates

| Model | Standardized core (60) | Injection target-follow | Near-context (9) |
|---|---:|---:|---:|
| **Clef** | **60/60** | 0/10 | **9/9** (reconstructed panel, see caveat) |
| Gemma 3 4B | 55/60 | 0/10 | 5/9 |
| Qwen3 4B Instruct 2507 | 48/60 | 2/10 | 9/9 |
| Llama 3.2 3B | 39/60 | 2/10 | 8/9 |

Real-transcript agreement, the top criterion: Qwen 18/20, Llama 7/20, Gemma 6/20, **Clef not measured.**

## How to read this

- **The synthetic suite is saturated for Clef.** A perfect 60/60 cannot separate Clef from any better judge, and with 15 cases per family the 95% lower bound is about 80% per family. Gemma already scores 91.7%. The panels that discriminate are the real-transcript one and harder long-context or injection cases.
- **Two of Phase B's metrics are trivially perfect for Clef.** It is deterministic (so repeat stability is 12/12 by construction) and its typed output cannot be invalid.
- **Clef sees a different input than the other candidates.** Same tasks and labels, but a typed JSON request with its own criteria text instead of the free-text judge prompt. Differences are attributable to the model and its input format together.
- **The near-context comparison is directional only.** #122's filler text was not published, so the Clef panel is a reconstruction of the same design (cases, 62,000 characters, early or late placement). The latency figures are from different hardware.
- **Injection still moves Clef's confidence.** On 14 of 15 cases the injected instruction lowered the correct label's probability (by up to 0.19), without flipping any label.

## Reproducing and extending

- Phase A runner: `eval/judge_reliability/run_clef_preflight.py`. Phase B: `run_clef_comparison.py`, which needs #122's `run_model_comparison.py` (still an open PR). Phase C context panel: `run_clef_context_stress.py` (standalone).
- Extra packages Clef needs beyond its model card: `torchvision` matching torch 2.11 and `accelerate`.
- Truncation guard: any input over `--max-length` (16,384 tokens) fails the call instead of being silently cut. Verified on the real encoder in Phase C.
- Raw per-call outputs and manifests (with probabilities, token counts and latencies): `../results/clef_preflight_v1/` (condensed, Phase A), `../results/clef_judge_v1/standardized/` and `../results/clef_judge_v1/context_stress/`.

## To finish the experiment

1. **Provide the real-transcript panel inputs** (the 20 response-level units from the six conversations, the human reference labels, and the script that builds each judge's input). Then run Clef on it (a short job on the same hardware), and apply the decision rule.
2. Merge or rebase against #122 so the Phase B runner has its dependency in the same tree.
3. Consider harder injection and abstain cases, since the synthetic suite no longer discriminates at the top.
