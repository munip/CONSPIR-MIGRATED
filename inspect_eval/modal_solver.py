"""Run the ConspiracyBench solver on Modal against a vLLM server, for the ~4B roster.

    modal run inspect_eval/modal_solver.py --model qwen3.5-4b --limit 3 --epochs 1                 # smoke
    modal run --detach inspect_eval/modal_solver.py --model qwen3.5-4b \
        --variants multiturn,multiturn_first,single_turn --tag full-2026-10-23                       # baseline
    modal volume get cb-judges-results solver/<model>/<tag> runs/

One container per model: vLLM serves the model on one GPU, and Inspect runs each variant against it
with ``--no-score``, all variants at once, so the transcripts are graded afterwards by the deployed
decision judges (``export_units.py`` -> ``decision_judges/modal_grade.py`` -> ``decision_report.py``).

Everything is kept on the ``cb-judges-results`` volume under ``solver/<model>/<tag>/``: one Inspect log
per variant (every message, reasoning, per-turn metadata and model call), each Inspect process's
output, the vLLM server log, ``manifest.json`` (roster entry, settings, commit, timings, exit
statuses), ``pip_freeze.txt`` and ``nvidia_smi.txt``. The volume is committed every minute, and the
container prints progress with an example answer every minute while it runs.

Every model thinks (decided 2026-10-23): thinking is switched on explicitly per request through
``chat_template_kwargs``, including Gemma 4, which doesn't think by default, and K2 Horizon runs at
the high reasoning effort its card asks evaluations to use. Every model samples with its own
recommended settings: vLLM loads them from the repo's ``generation_config.json`` (Gemma 4, Granite,
Nemotron), and for the two repos without one (Qwen3.5 4B, K2 Horizon) the model card's settings
are sent with every request (``sampling``). Every model gets the same 8,192-token
reply cap and 32k context budget, so a k = 5 run rarely needs the sliding window. Reasoning is split from the answer by
each model's vLLM reasoning parser, so the judge only sees the answer.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_TOKENS = 32768
MAX_TOKENS = 8192  # README "Speed test": at 2048 Qwen3.5 4B's thinking left half its replies empty or cut off

# key -> HF repo, vLLM arguments, and notes. Same-size roster agreed 2026-10-23 (Spark dropped;
# Ling 3.0 Tiny / Ling 2.0 Mini are 8B / 16B-total MoE, so not comparable).
# "template" is sent as chat_template_kwargs on every request. Parsers are vLLM 0.32's own, except
# Nemotron's, which its card ships as a plugin.
THINK = {"enable_thinking": True}
ROSTER = {
    # No generation_config.json in the repo, so its card's thinking-mode settings are sent explicitly.
    # Without presence_penalty the smoke test looped ("Okay." x173) until max_tokens on 2 of 12 turns.
    "qwen3.5-4b": {"repo": "Qwen/Qwen3.5-4B", "args": ["--reasoning-parser", "qwen3"], "template": THINK,
                   "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                "presence_penalty": 1.5, "repetition_penalty": 1.0}},
    "gemma-4-e4b": {"repo": "google/gemma-4-E4B-it", "args": ["--reasoning-parser", "gemma4"], "template": THINK},
    "granite-4.2-3b": {"repo": "ibm-granite/granite-4.2-3b", "args": ["--reasoning-parser", "granite_thinking_parser"],
                       "template": THINK},
    "nemotron-3-nano-4b": {"repo": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "plugin": "nano_v3_reasoning_parser.py",
                           "args": ["--reasoning-parser", "nano_v3", "--trust-remote-code"], "template": THINK},
    # The card asks for reasoning_effort "high" with at least 32k output tokens; it runs at "high" but
    # with the same 8,192-token cap as the rest of the roster (decided 2026-10-23), so its cut-off
    # replies are invalid like any other model's.
    # No generation_config.json either: its card's recommended settings are sent explicitly.
    "k2-horizon-3.7b": {"repo": "IFM/K2-Horizon-3.7B", "args": ["--reasoning-parser", "k2_horizon", "--trust-remote-code"],
                        "template": {"reasoning_effort": "high"}, "sampling": {"temperature": 1.0, "top_p": 0.95}},
}

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.32.0", "inspect-ai==0.3.283", "openai", "jsonschema", "huggingface_hub")
         # The image has no CUDA toolkit, so FlashInfer's sampler can't JIT-compile its top-k / top-p
         # kernel (the smoke test failed on it); vLLM's PyTorch sampler needs no compiler.
         .env({"HF_HOME": "/hf", "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1",
               "VLLM_USE_FLASHINFER_SAMPLER": "0"})
         .add_local_dir(ROOT / "inspect_eval", "/repo/inspect_eval", ignore=["**/__pycache__/**", "logs/**"])
         .add_local_dir(ROOT / "eval", "/repo/eval", ignore=["**/__pycache__/**", "judge_reliability/**"])
         .add_local_dir(ROOT / "questions", "/repo/questions")
         .add_local_file(ROOT / "schema.json", "/repo/schema.json")
         .add_local_file(ROOT / "completed_experiments" / "judge_reliability" / "run_baseline.py",
                         "/repo/completed_experiments/judge_reliability/run_baseline.py"))
hf = modal.Volume.from_name("cb-eval-hf", create_if_missing=True)
results = modal.Volume.from_name("cb-judges-results", create_if_missing=True)
app = modal.App("cb-eval-solver")


def _serve(spec: dict, log) -> subprocess.Popen:
    args = list(spec["args"])
    if spec.get("plugin"):
        from huggingface_hub import hf_hub_download

        args += ["--reasoning-parser-plugin", hf_hub_download(spec["repo"], spec["plugin"])]
    cmd = ["vllm", "serve", spec["repo"], "--served-model-name", "cb",
           "--max-model-len", str(spec.get("context_tokens", CONTEXT_TOKENS)),
           "--port", "8000", *args]
    print(" ".join(cmd), flush=True)
    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(180):  # up to 30 minutes for download + load
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=5)
            return server
        except Exception:
            if server.poll() is not None:
                raise RuntimeError(f"vLLM exited with {server.returncode}; see vllm_server.log")
            time.sleep(10)
    raise TimeoutError("vLLM did not come up in 30 minutes")


def _progress(model: str, logs: dict) -> None:
    """Print each variant's completed samples, and one recent question with the model's answer."""
    from inspect_ai.log import read_eval_log, read_eval_log_sample, read_eval_log_sample_summaries

    example = None
    for variant, log_dir in logs.items():
        files = sorted(log_dir.glob("*.eval"))
        if not files:
            print(f"[progress] {model} {variant}: starting", flush=True)
            continue
        try:
            header = read_eval_log(str(files[-1]), header_only=True)
            planned = (header.eval.dataset.samples or 0) * (header.eval.config.epochs or 1)
            done = [s for s in read_eval_log_sample_summaries(str(files[-1])) if s.completed]
            errors = sum(1 for s in done if s.error)
            print(f"[progress] {model} {variant}: {len(done)}/{planned} samples done, {errors} errored", flush=True)
            if done and (example is None or variant == "multiturn"):
                example = (variant, files[-1], done[-1])
        except Exception as exc:  # a log mid-write; tried again next minute
            print(f"[progress] {model} {variant}: unreadable right now ({type(exc).__name__})", flush=True)
    if example:
        variant, path, summary = example
        try:
            sample = read_eval_log_sample(str(path), summary.id, summary.epoch)
            users = [m.text for m in sample.messages if m.role == "user"]
            answers = [m.text for m in sample.messages if m.role == "assistant"]
            print(f"[example] {model} {variant} {sample.id} epoch {sample.epoch}, {len(answers)} turn(s)", flush=True)
            print(f"[example] Q1: {users[0][:300]!r}", flush=True)
            print(f"[example] A1: {(answers[0] if answers else '')[:500]!r}", flush=True)
            if len(users) > 1:
                print(f"[example] last pressure turn: {users[-1][:300]!r}", flush=True)
                print(f"[example] last answer: {(answers[-1] if answers else '')[:500]!r}", flush=True)
        except Exception as exc:
            print(f"[example] unavailable right now ({type(exc).__name__})", flush=True)


# "H100!" stops Modal upgrading a call to an H200, as for the judge (decision_judges/modal_grade.py).
@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/hf": hf, "/results": results}, timeout=24 * 3600)
def solve(model: str, variants: list[str], epochs: int, limit: int | None, max_connections: int, tag: str,
          suite_commit: str) -> dict:
    spec = ROSTER[model]
    base = Path("/results/solver") / model / tag
    base.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = {"model": model, "roster_entry": spec, "variants": variants, "epochs": epochs, "limit": limit,
                "max_connections": max_connections, "max_tokens": spec.get("max_tokens", MAX_TOKENS),
                "context_tokens": spec.get("context_tokens", CONTEXT_TOKENS), "suite_commit": suite_commit,
                "gpu": "H100!", "modal_task_id": os.environ.get("MODAL_TASK_ID"),
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (base / "pip_freeze.txt").write_text(subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout)
    (base / "nvidia_smi.txt").write_text(subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True).stdout)
    server_log = open(base / "vllm_server.log", "a")
    server = _serve(spec, server_log)
    hf.commit()
    manifest["server_ready_seconds"] = round(time.time() - started)
    print(f"[progress] {model}: vLLM ready after {manifest['server_ready_seconds']} s; running {', '.join(variants)}",
          flush=True)
    extra_body = json.dumps({"chat_template_kwargs": spec["template"], **spec.get("sampling", {})})
    env = {**os.environ, "VLLM_BASE_URL": "http://localhost:8000/v1", "VLLM_API_KEY": "none"}
    procs, logs = {}, {}
    for variant in variants:
        logs[variant] = base / variant
        # Relative to cwd=/repo: Inspect globs the task path, and Python 3.12 rejects absolute globs.
        cmd = ["inspect", "eval", "inspect_eval/conspiracybench.py", "--model", "openai-api/vllm/cb", "--no-score",
               "-T", f"epochs={epochs}", "-T", f"variant={variant}", "-T", f"max_tokens={manifest['max_tokens']}",
               "-T", f"context_tokens={manifest['context_tokens']}", "--extra-body", extra_body,
               "--max-connections", str(max_connections), "--retry-on-error", "3", "--attempt-timeout", "1800",
               "--log-buffer", "5", "--log-dir", str(logs[variant]), "--display", "plain"]
        if limit:
            cmd += ["--limit", str(limit)]
        out = open(base / f"inspect_{variant}.log", "a")
        procs[variant] = (subprocess.Popen(cmd, env=env, cwd="/repo", stdout=out, stderr=subprocess.STDOUT), out)
    while any(p.poll() is None for p, _ in procs.values()):
        time.sleep(60)
        _progress(model, logs)
        results.commit()
    _progress(model, logs)
    manifest["exit_status"] = {v: p.returncode for v, (p, _) in procs.items()}
    manifest["minutes"] = round((time.time() - started) / 60, 1)
    manifest["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    server.terminate()
    server_log.close()
    results.commit()
    return {"model": model, "variants": variants, "exit_status": manifest["exit_status"],
            "minutes": manifest["minutes"], "dir": str(base)}


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/hf": hf, "/results": results}, timeout=24 * 3600)
def retry_errored(model: str, tag: str, variant: str, suite_commit: str) -> dict:
    """Re-run only the errored samples of a variant's latest log (``inspect eval-retry``).

    The retried log is written beside the original, which is kept; it holds every sample, the
    completed ones copied from the original. Recorded under ``retries`` in the run's manifest.
    """
    spec = ROSTER[model]
    base = Path("/results/solver") / model / tag
    original = sorted((base / variant).glob("*.eval"))[-1]
    server_log = open(base / "vllm_server_retry.log", "a")
    server = _serve(spec, server_log)
    env = {**os.environ, "VLLM_BASE_URL": "http://localhost:8000/v1", "VLLM_API_KEY": "none"}
    started = time.time()
    # --no-score again: eval-retry doesn't carry it over, and without it Inspect asks for the grader role.
    cmd = ["inspect", "eval-retry", str(original), "--no-score", "--log-dir", str(base / variant), "--max-connections", "32",
           "--retry-on-error", "3", "--log-buffer", "5", "--display", "plain"]
    with open(base / f"inspect_{variant}_retry.log", "a") as out:
        status = subprocess.run(cmd, env=env, cwd="/repo", stdout=out, stderr=subprocess.STDOUT).returncode
    server.terminate()
    server_log.close()
    manifest = json.loads((base / "manifest.json").read_text())
    manifest.setdefault("retries", []).append({
        "variant": variant, "original_log": original.name, "suite_commit": suite_commit, "exit_status": status,
        "minutes": round((time.time() - started) / 60, 1), "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results.commit()
    return manifest["retries"][-1]


@app.local_entrypoint()
def main(model: str = "qwen3.5-4b", variants: str = "multiturn", epochs: int = 5, limit: int = 0,
         max_connections: int = 32, tag: str = "", retry: str = ""):
    if model not in ROSTER:
        raise SystemExit(f"unknown model {model!r}; roster: {', '.join(ROSTER)}")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if retry:  # --retry <variant> --tag <run>: re-run that variant's errored samples
        print(retry_errored.remote(model, tag, retry, commit))
        return
    tag = tag or ("smoke" if limit else time.strftime("full-%Y%m%dT%H%MZ", time.gmtime()))
    picked = [v.strip() for v in variants.split(",") if v.strip()]
    print(solve.remote(model, picked, epochs, limit or None, max_connections, tag, commit))
