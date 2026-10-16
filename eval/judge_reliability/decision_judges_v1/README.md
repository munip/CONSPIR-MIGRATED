# Decision judges v1: one GPU run of Clef, Kev-4B, Laya and pplx-decider

The GPU run asked for in #203 and #208: the four decision-model candidates on the 150-unit
real-transcript validity set (244 calls each), under one fixed setup, logging both validity and
speed. `protocol.json` holds every fixed choice; this file is the how-to.

## What is held the same for every judge

- **Hardware:** a Modal container per judge, all of one spec: `H100!` (NVIDIA H100 80GB HBM3, never
  silently upgraded to an H200), 8 CPUs, 128 GB RAM, driver 610.57.04. Each run refuses to start if
  another process holds its GPU and saves `nvidia-smi -q`.
- **Software:** one locked environment (`env/uv.lock`, installed from `env/requirements.txt` with
  `--no-deps`): Python 3.12, torch 2.14.0 (CUDA 13), transformers 5.17.0, flash-linear-attention 0.5.2.
  Kev declares `torch<2.9`; the env overrides that, and Kev passed its smoke test under it, so all four
  ran in the same environment (`env-kev/`, torch 2.8.0, was not needed). Each run saves `pip_freeze.txt`.
- **Inputs:** the same 244 requests from `run_decision_validity.py` (units sha256 in the manifest), in the
  same order, batch size 1, no truncation for any judge.
- **Process:** one fresh process per judge (pplx changes process-wide torch settings), models
  downloaded before any timed run, and the same 3 untimed warm-up calls first.
- **Measurement:** per-call latency, input tokens, model load and warm-up per session, peak GPU memory,
  and nvidia-smi samples every second (utilisation, memory, power, clock, temperature).

## Why an 80 GB GPU

Clef (55 GB) and pplx-decider (52 GB) are bf16 27B models, and both loaders place the whole model on a
single device, so every judge needs one GPU with 80 GB or more for the four to share one setup.

## Run it on Modal (how v1 was run)

```bash
modal run modal_app.py::download   # all four models to volume cb-judges-hf, at their pinned revisions
modal run modal_app.py::smoke      # 3 calls per judge
modal run modal_app.py::full       # 244 calls per judge, one container each
modal volume get cb-judges-results / ../results/
python speed_report.py --price-per-hour 5.35 ../results/{clef,kev,laya,pplx}_validity_v1
```

$5.35/h is the whole container at Modal's list prices (H100 $3.95/h + 8 cores + 128 GiB); see
`protocol.json`.

## Run it on a self-hosted box (Vultr)

Vultr was the first plan. Its only ≥80 GB GPU in stock, a GH200 (`vbm-72c-480gb-gh200-gpu`, Frankfurt,
$1.99/h, preemptible only, aarch64), needs a $50 cash deposit before it deploys, so v1 moved to Modal.
These scripts still work for any SSH-reachable box with a ≥80 GB GPU:

```bash
# 0. Your machine. The Vultr key's IP allow-list must include this machine's IP.
export VULTR_API_KEY=$(security find-generic-password -s vultr-api-key -w)
python vultr.py stock && python vultr.py os && python vultr.py sshkeys
python vultr.py create --os-id <arm64 ubuntu id> --sshkey-id <id> --yes     # starts billing
python vultr.py status <id>                                                  # wait for active + IP

# 1. Ship the committed branch (a git bundle, nothing is pushed) and set the host up.
./deploy.sh root@<ip>

# 2. On the host: smoke test all four (3 calls each), then the full runs.
ssh root@<ip> "tmux new -d -s judges 'bash cb/eval/judge_reliability/decision_judges_v1/run_all.sh 2>&1 | tee -a judges.log'"
#    If only Kev's smoke test fails under the torch override:
#    rm -r results/kev_validity_v1_smoke; KEV_ENV=env-kev bash .../run_all.sh   (and note it in the report)

# 3. Your machine, during and after the run (the box is preemptible):
while sleep 300; do ./pull_results.sh root@<ip>; done

# 4. When all four are done and pulled: destroy the box.
python vultr.py destroy <id> --yes
```

Results land in `eval/judge_reliability/results/<judge>_validity_v1/` (`results.jsonl`,
`sessions.jsonl`, `manifest.json`, `gpu_telemetry.csv`, `pip_freeze.txt`). The speed table is in
`results/decision_judges_v1_speed/`. Rebuild it any time with:

```bash
python speed_report.py --price-per-hour 1.99 ../results/{clef,kev,laya,pplx}_validity_v1
```

## Validity, once the final labels are in

Scoring is the pre-registered `score_validity.py`; it needs no GPU. The reference is `reference_v1.csv`
(#364: all 150 units, 111 hold / 34 diverge / 5 abstain). Every reference label is a human's. On
u021–u075 the second labeller was Qwen3-32B, and every disagreement went to the human (powercoding1), so
that slice rests on one person rather than two; the report shows it separately:

```bash
python eval/judge_reliability/score_validity.py \
  --units eval/judge_reliability/validity_v1/units.jsonl \
  --labels eval/judge_reliability/validity_v1/labels/sheet_*.csv \
  --adjudicated eval/judge_reliability/validity_v1/labels/reference_v1.csv \
  --class-mix <the real-run class mix used for the chat judges> \
  --pred eval/judge_reliability/results/{clef,kev,laya,pplx}_validity_v1/results.jsonl \
  --out eval/judge_reliability/results/decision_judges_v1_validity.json
```

## Cost (v1, actual)

Each full judge run took 1–4 minutes of container time on Modal: about $0.09–0.30 per judge at
$5.35/h, $0.86 for all four. Downloads ran on a CPU container, and the smoke test added about as much
again. Results and the write-up: `../experiments/decision_judges_v1_report.md`.
