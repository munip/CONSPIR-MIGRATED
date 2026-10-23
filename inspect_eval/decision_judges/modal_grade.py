"""Grade exported eval turns with pplx-decider on Modal, on one H100 (Clef optional, in parallel).

    python inspect_eval/export_units.py logs/<run>.eval --out units/<model>.jsonl
    modal run inspect_eval/decision_judges/modal_grade.py --units units/<model>.jsonl --tag <model> [--limit 3]
    modal volume get cb-judges-results eval/<tag> graded/

The container is the reliability study's (``decision_judges_v1/env``, GPU "H100!", models read
from the ``cb-judges-hf`` volume where they are already downloaded), so the judges run exactly
as they were validated. The units file is uploaded with the call and stored beside the results
in ``cb-judges-results`` under ``eval/<tag>/``. Results are committed every 30 s and a rerun with
the same tag resumes. ``--limit 3`` is a smoke test.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# In the container this file sits at /root, so the repo is found at /repo, where the image puts it.
ROOT = HERE.parents[1] if modal.is_local() else Path("/repo")
KIT = ROOT / "completed_experiments" / "judge_reliability"
GPU, CPU, MEMORY_MB = "H100!", 8, 131072
# pplx-decider is the benchmark's judge (decided 2026-10-23); add Clef with --judges pplx,clef.
JUDGES = ("pplx",)

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git")
         .pip_install("uv==0.12.5")
         .add_local_file(KIT / "decision_judges_v1" / "env" / "requirements.txt", "/tmp/requirements.txt", copy=True)
         .run_commands("uv pip install --system --no-deps -r /tmp/requirements.txt")
         .env({"HF_HOME": "/hf", "HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
         .add_local_dir(KIT, "/repo/completed_experiments/judge_reliability",
                        ignore=["results/**", "**/__pycache__/**", "**/.venv/**", "*/page/**"])
         .add_local_dir(ROOT / "eval", "/repo/eval", ignore=["**/__pycache__/**", "judge_reliability/**"])
         .add_local_dir(HERE, "/repo/inspect_eval/decision_judges", ignore=["**/__pycache__/**"]))
hf = modal.Volume.from_name("cb-judges-hf")
results = modal.Volume.from_name("cb-judges-results", create_if_missing=True)
app = modal.App("cb-eval-decision-judges")


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_MB, volumes={"/hf": hf, "/results": results}, timeout=8 * 3600)
def grade(judge: str, units_text: str, tag: str, limit: int | None, suite_commit: str) -> dict:
    base = Path("/results/eval") / tag
    out = base / judge
    out.mkdir(parents=True, exist_ok=True)
    units = base / "units.jsonl"
    # An empty units_text means the file was uploaded to the volume beforehand (modal volume put), as
    # a full run's units (60-100 MB per model) are too big to send as an argument.
    if units_text and (not units.exists() or units.read_text(encoding="utf-8") != units_text):
        units.write_text(units_text, encoding="utf-8")
    planned = sum(1 for line in units.read_text(encoding="utf-8").splitlines() if line.strip() and not json.loads(line).get("skip"))
    stop = threading.Event()

    def committer():  # keep the volume current and say how far the judge has got
        while not stop.wait(60):
            done = sum(1 for _ in (out / "results.jsonl").open()) if (out / "results.jsonl").exists() else 0
            print(f"[grading] {tag} {judge}: {done}/{min(planned, limit or planned)} graded", flush=True)
            results.commit()

    threading.Thread(target=committer, daemon=True).start()
    cmd = ["python", "/repo/inspect_eval/decision_judges/grade_units.py", "--judge", judge, "--units", str(units),
           "--output-dir", str(out), "--run"] + (["--limit", str(limit)] if limit else [])
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out / "run.log", "a") as log:
        log.write(f"\n=== {started} suite {suite_commit} {' '.join(cmd)}\n")
        log.flush()
        status = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()] if (out / "results.jsonl").exists() else []
    info = {"judge": judge, "tag": tag, "suite_commit": suite_commit, "started": started,
            "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "exit_status": status,
            "rows": len(rows), "errors": sum(r.get("status") != "ok" for r in rows)}
    with open(out / "run_info.jsonl", "a") as f:
        f.write(json.dumps(info) + "\n")
    stop.set()
    results.commit()
    return info


@app.local_entrypoint()
def main(units: str, tag: str, judges: str = ",".join(JUDGES), limit: int = 0, uploaded: bool = False):
    """``--uploaded``: ``units`` was already put on the volume at eval/<tag>/units.jsonl; don't send it."""
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    text = "" if uploaded else Path(units).read_text(encoding="utf-8")
    picked = [j for j in judges.split(",") if j]
    for info in grade.starmap([(j, text, tag, limit or None, commit) for j in picked], return_exceptions=True):
        print(json.dumps(info, default=str))
