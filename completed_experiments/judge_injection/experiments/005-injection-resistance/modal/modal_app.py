"""Run Experiment 005 on Modal: RUN.md's steps 6-8 in one H100 container.

    modal run --detach modal_app.py::full
    modal volume get cb-exp005-results / <local dir>

One image built from env/uv.lock (requirements.txt exported with `uv export --frozen`, installed with
--no-deps), GPU "H100!" (no H200 upgrade), 8 CPUs, 128 GB RAM, model weights on the cb-judges-hf volume
(Clef, pplx, Kev and Laya at the pinned revisions are already there from the decision-judge run). Each judge runs
as its own process, one after another: a 3-row smoke run, then --resume for the rest. Then the cascade
and an --inspect-only completion check. Results commit to the volume every 30 s; rerunning resumes.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import modal

KIT = Path(__file__).resolve().parent
REMOTE = "/repo"
REPO = KIT.parents[4] if modal.is_local() else Path(REMOTE)
RUNNER = f"{REMOTE}/completed_experiments/judge_injection/experiments/005-injection-resistance/experiment.py"
OUT = Path("/results/exp005")
GPU, CPU, MEMORY_MB = "H100!", 8, 131072

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git")
         .pip_install("uv==0.12.5")
         .add_local_file(KIT / "requirements.txt", "/tmp/requirements.txt", copy=True)
         .run_commands("uv pip install --system --no-deps -r /tmp/requirements.txt")
         .env({"HF_HOME": "/hf", "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
         .add_local_dir(REPO, REMOTE, ignore=[".git", ".git/**", ".claude/**", "**/__pycache__/**", "**/.venv/**",
                                              "completed_experiments/judge_injection/experiments/**",
                                              "completed_experiments/judge_injection/reports/**",
                                              "completed_experiments/judge_injection/experiments/005-injection-resistance/modal/**"]))
hf = modal.Volume.from_name("cb-judges-hf", create_if_missing=True)
results = modal.Volume.from_name("cb-exp005-results", create_if_missing=True)
app = modal.App("cb-exp005-injection-resistance")


def _runner(log, *args: str) -> int:
    cmd = ["python", RUNNER, *args, "--output-dir", str(OUT)]
    log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(cmd)}\n")
    log.flush()
    status = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    log.write(f"=== exit {status}\n")
    log.flush()
    return status


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_MB, volumes={"/hf": hf, "/results": results},
              timeout=6 * 3600)
def run_all(source_commit: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "source-commit.txt").write_text(source_commit + "\n")
    (OUT / "nvidia_smi.txt").write_text(subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True).stdout)
    stop = threading.Event()

    def committer():
        while not stop.wait(30):
            results.commit()

    threading.Thread(target=committer, daemon=True).start()
    statuses: dict[str, list[int]] = {}
    with open(OUT / "run.log", "a") as log:
        for judge in ("qwen", "pplx", "clef", "kev", "laya"):
            done = []
            if not (OUT / f"{judge}.results.jsonl").exists():
                done.append(_runner(log, "--run", "--judge", judge, "--limit", "3"))
                if done[-1] != 0:
                    statuses[judge] = done
                    break
            done.append(_runner(log, "--resume", "--judge", judge))
            statuses[judge] = done
            hf.commit()  # keep downloaded Qwen weights for reruns
        else:
            cascade = "--resume" if (OUT / "cascade.results.jsonl").exists() else "--run"
            statuses["cascade"] = [_runner(log, cascade, "--judge", "cascade")]
        statuses["inspect"] = [_runner(log, "--inspect-only", "--judge", "qwen")]
    (OUT / "pip_freeze.txt").write_text(subprocess.run(["uv", "pip", "freeze", "--system"],
                                                       capture_output=True, text=True).stdout)
    stop.set()
    results.commit()
    summary = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else None
    return {"statuses": statuses, "summary": summary}


@app.local_entrypoint()
def full():
    commit = subprocess.run(["git", "-C", str(KIT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(json.dumps(run_all.remote(commit), indent=2))
