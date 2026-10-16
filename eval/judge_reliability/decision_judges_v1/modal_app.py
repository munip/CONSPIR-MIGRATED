"""Run the four decision judges on Modal under one fixed setup (protocol.json).

    modal run modal_app.py::download          # fetch all four models to a volume (CPU, untimed)
    modal run modal_app.py::smoke             # 3 calls per judge
    modal run modal_app.py::full              # all 244 calls per judge
    modal volume get cb-judges-results / <local dir>

Every judge gets the same container: one image built from env/uv.lock (installed with --no-deps, so
the locked versions are exactly what runs, including the torch override for Kev), GPU type "H100!"
(the "!" stops Modal upgrading a call to an H200), 8 CPUs and 128 GB RAM, models read from the same
volume, and a fresh container per judge. Results are written to a volume as they happen and
committed every 30 s, so a lost container loses at most that much; a rerun resumes.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

KIT = Path(__file__).resolve().parent
REMOTE = "/repo/suite"  # the repo's top-level test-suite directory (this file's grandparent) is mounted here
SUITE = KIT.parents[1] if modal.is_local() else Path(REMOTE)  # in the container this file sits at /root
GPU, CPU, MEMORY_MB = "H100!", 8, 131072
MODELS = [
    ("Cloudflare/clef", "2f3de3dd85f379784083b0814d997ab627200f0c"),
    ("jaredpalmer/kev-4b", "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"),
    ("convaiinnovations/laya", "7b928d828b7b0e022f929d9bd2e44165aa270148"),
    ("perplexity-ai/pplx-decider-v1.1-27b", "14c9d90137962d300f3de95633f7b6535e80c5ce"),
]
JUDGES = ("clef", "kev", "laya", "pplx")


def image_for(env: str) -> modal.Image:
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("git")
            .pip_install("uv==0.12.5")
            .add_local_file(KIT / env / "requirements.txt", "/tmp/requirements.txt", copy=True)
            .run_commands("uv pip install --system --no-deps -r /tmp/requirements.txt")
            .env({"HF_HOME": "/hf", "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
            .add_local_dir(SUITE, REMOTE, ignore=["judge_injection/**", "**/results/**", "**/__pycache__/**", "**/.venv/**"]))


image, image_kev = image_for("env"), image_for("env-kev")
hf = modal.Volume.from_name("cb-judges-hf", create_if_missing=True)
results = modal.Volume.from_name("cb-judges-results", create_if_missing=True)
app = modal.App("cb-decision-judges-v1")


@app.function(image=image, volumes={"/hf": hf}, cpu=8, memory=32768, timeout=3 * 3600)
def fetch_models() -> list[str]:
    from huggingface_hub import snapshot_download

    paths = [f"{repo}@{rev[:8]} -> {snapshot_download(repo, revision=rev)}" for repo, rev in MODELS]
    hf.commit()
    return paths


def _run(judge: str, stage: str, suite_commit: str, env: str) -> dict:
    out = Path("/results") / (f"{judge}_validity_v1" + ("_smoke" if stage == "smoke" else ""))
    out.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / f"nvidia_smi_{started}.txt").write_text(subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True).stdout)
    busy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    if busy:
        raise RuntimeError(f"GPU is busy: {busy}")

    stop = threading.Event()

    def committer():  # keep the volume current while the judge runs
        while not stop.wait(30):
            results.commit()

    threading.Thread(target=committer, daemon=True).start()
    telemetry = open(out / "gpu_telemetry.csv", "a")
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu",
                            "--format=csv,nounits", "-l", "1"], stdout=telemetry)
    cmd = ["python", f"{REMOTE}/judge_reliability/run_decision_validity.py", "--judge", judge,
           "--units", f"{REMOTE}/judge_reliability/validity_v1/units.jsonl", "--output-dir", str(out),
           "--device", "cuda", "--warmup", "3", "--run"] + (["--limit", "3"] if stage == "smoke" else [])
    with open(out / "run.log", "a") as log:
        log.write(f"\n=== {started} {' '.join(cmd)}\n")
        log.flush()
        status = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    smi.terminate()
    telemetry.close()
    (out / "pip_freeze.txt").write_text(subprocess.run(["uv", "pip", "freeze", "--system"], capture_output=True, text=True).stdout)
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()] if (out / "results.jsonl").exists() else []
    info = {"judge": judge, "stage": stage, "env": env, "suite_commit": suite_commit, "gpu_request": GPU, "cpu": CPU,
            "memory_mb": MEMORY_MB, "modal_task_id": os.environ.get("MODAL_TASK_ID"), "started": started,
            "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "exit_status": status,
            "rows": len(rows), "errors": sum(r.get("status") != "ok" for r in rows)}
    with open(out / "run_info.jsonl", "a") as f:
        f.write(json.dumps(info) + "\n")
    stop.set()
    results.commit()
    hf.commit()  # Kev resolves its base model on first load
    return info


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_MB, volumes={"/hf": hf, "/results": results}, timeout=4 * 3600)
def run_judge(judge: str, stage: str, suite_commit: str) -> dict:
    return _run(judge, stage, suite_commit, "env")


@app.function(image=image_kev, gpu=GPU, cpu=CPU, memory=MEMORY_MB, volumes={"/hf": hf, "/results": results}, timeout=4 * 3600)
def run_kev_torch28(stage: str, suite_commit: str) -> dict:
    return _run("kev", stage, suite_commit, "env-kev")


def _commit() -> str:
    return subprocess.run(["git", "-C", str(KIT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def _stage(stage: str, judges: str) -> None:
    commit = _commit()
    picked = [j for j in judges.split(",") if j]
    for info in run_judge.starmap([(j, stage, commit) for j in picked], return_exceptions=True):
        print(json.dumps(info, default=str))


@app.local_entrypoint()
def download():
    for line in fetch_models.remote():
        print(line)


@app.local_entrypoint()
def smoke(judges: str = ",".join(JUDGES)):
    _stage("smoke", judges)


@app.local_entrypoint()
def full(judges: str = ",".join(JUDGES)):
    _stage("full", judges)


@app.local_entrypoint()
def kev_fallback(stage: str = "smoke"):
    print(json.dumps(run_kev_torch28.remote(stage, _commit())))
