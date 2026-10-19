# Validity set v1: chat judges, jury and cascade (#203)

Companion to Laura's decision-model run (#365). Same 150 units (#208), same reference (#364:
111 hold / 34 diverge / 5 abstain), same pre-registered `score_validity.py`. The full results
and discussion are on #203. This file records what was run and how to reproduce it.

## What was run
- **Chat judges:** `inspect_eval/judge_cases.py -T cases=completed_experiments/judge_reliability/validity_v1/units.jsonl`, the exact prompt the Inspect scorer sends, at temperature 0 and `--max-connections 1`. All on Ollama, at `num_ctx` 16384.
  - `cb-qwen3-4b` (qwen3:4b-instruct-2507, the current judge), `cb-gemma3-4b`, `cb-llama3.2-3b` and `cb-qwen3.5-9b` (thinking off via `--extra-body '{"reasoning_effort":"none"}'`): local, Apple M3 Pro, Ollama 0.35.1.
  - `cb-gpt-oss-20b`: on a Freestyle VM (8 vCPU, 32 GB, CPU only, Ollama 0.40.2 with `LLAMA_ARG_CACHE_RAM=2048`). Median 54 s/unit, 2.5 h in total.
  - All 150 prompts fit (max 4,746 tokens). A first Qwen3.5 9B run at Ollama's default 4096 context truncated ~9 units, so it was discarded and rerun at 16384.
- **The jury, pre-stated on Discord before any labels:** Gemma 3 4B + Llama 3.2 3B + Clef (no context), with a strict majority.
- **Cascades:**
  - Clef → Qwen3-4B (pre-stated in #203).
  - pplx → Qwen3-4B (exploratory: chosen after #365 was public).
  
  τ is chosen by 5-fold cross-validation (#323). It was 0.35 in every fold for pplx.

## Files
- `results/chat_judges_validity_v1/*.eval`: the raw Inspect logs, one per chat judge.
- `results/validity_v1_scoring/results_ref_all.json`, `results_ref_twohuman.json`: the full `score_validity.py --out` output.
  - Every judge and scope: all units, opening and later turns, per source model, and per input size where recorded.
  - Every metric: recall with Wilson intervals, raw and reweighted precision, judge-vs-human rates, κ.
- `results/validity_v1_scoring/ref_twohuman.csv`: the 95 units with two human labels (u001–u020, u076–u150).
- `validity_v1/key.csv` and `validity_v1/solver_logs/*.eval`: the unit→model key and the two target solver logs (Gemma 3 4B; Qwen3.5 4B with thinking off), held back during labelling (#208) and published now that labelling is done.

## Reproduce
```bash
python completed_experiments/judge_reliability/score_validity.py --units completed_experiments/judge_reliability/validity_v1/units.jsonl \
  --labels completed_experiments/judge_reliability/validity_v1/labels/sheet_{powercoding1,divyanshijoshi,vaisakhi}.csv \
  --adjudicated completed_experiments/judge_reliability/validity_v1/labels/reference_v1.csv \
  --key completed_experiments/judge_reliability/validity_v1/key.csv \
  --pred completed_experiments/judge_reliability/results/chat_judges_validity_v1/*.eval \
         completed_experiments/judge_reliability/results/{clef,pplx,kev,laya}_validity_v1/results.jsonl \
  --jury prestated=ollama/cb-gemma3-4b,ollama/cb-llama3.2-3b,clef_validity_v1/no_context \
  --cascade clef_qwen3=clef_validity_v1/no_context,ollama/cb-qwen3-4b \
  --cascade pplx_qwen3_exploratory=pplx_validity_v1/no_context,ollama/cb-qwen3-4b \
  --class-mix hold=0.8,diverge=0.15,abstain=0.05
```
For the conservative slice, pass `--adjudicated results/validity_v1_scoring/ref_twohuman.csv`.
`reference_v1.csv` comes from #364 and the decision-model outputs from #365.

## Limits
- Only 5 units are abstain, so abstain recall can't be measured.
- On u021–u075 the second label was Qwen3-32B, with disagreements going to the human, so in practice that slice is one human labeller. On u001–u020 the adjudicator was also a labeller.
- One sample (k = 1) of one corpus snapshot and two target models.
