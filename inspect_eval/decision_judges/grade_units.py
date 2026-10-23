#!/usr/bin/env python3
"""Grade exported eval turns with a deployed decision judge (pplx-decider, or Clef), on a GPU.

    python inspect_eval/decision_judges/grade_units.py --judge pplx --units units.jsonl \
        --output-dir graded/<model>/pplx --run

Units come from ``inspect_eval/export_units.py``. Every unit gets the verdict exactly as the
reliability rerun on reference v1.2 asked it (pplx κ .755): the endorsement rules
(``--criteria rubric_v2``), and the whole conversation before the reply as ``preceding_exchange``
for a turn that answers a pressure message; the opening question goes without one
(``run_decision_validity.build_request``).

pplx reads at most 8,192 tokens and Clef 16,384. When a conversation doesn't fit, the oldest
exchanges are left out until it does (``dropped`` in the row says how many messages), keeping the
pressure message being answered; the item's question, ground truth and lists are never cut.

Rows go to ``results.jsonl`` as they finish, one per unit, and a rerun skips units already there,
so an interrupted run resumes. ``--inspect-only`` writes the plan without loading a model. Runs on
Modal through ``modal_grade.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "completed_experiments" / "judge_reliability"))

import run_clef_preflight as clef  # noqa: E402
from run_decision_validity import PPLX, build_request  # noqa: E402

CRITERIA = "rubric_v2"  # the endorsement rules


def unit_request(unit: dict, transcript: list, model_id: str) -> dict:
    """The judge request for one unit, with ``transcript`` as the context."""
    unit = {**unit, "transcript": transcript}
    return build_request(unit, "preceding_exchange" if transcript else "no_context", model_id, CRITERIA)


def plan(units: list[dict]) -> list[dict]:
    return [unit for unit in units if not unit.get("skip")]


def load_infer(args):
    """``infer(request) -> (choice, confidence, probabilities)``."""
    if args.judge == "pplx":
        from huggingface_hub import snapshot_download

        checkpoint = Path(snapshot_download(PPLX["repo"], revision=PPLX["revision"]))
        sys.path.insert(0, str(checkpoint / "source" / "src"))
        from autojev.model import DecisionModel

        model = DecisionModel(checkpoint, device=args.device)

        def infer(request):
            question = request["questions"]["verdict"]
            row = {"state": json.dumps(request["state"], ensure_ascii=False, indent=1), "question": question}
            raw = model.predict([row])[0]  # raises on inputs over its 8,192-token limit
            names = list(question["criteria"])
            values = [float(raw[n]) for n in names] if isinstance(raw, dict) else [float(v) for v in raw]
            probs = {n: v / sum(values) for n, v in zip(names, values)}
            choice = max(probs, key=probs.get)
            k = len(names)
            return choice, (probs[choice] - 1 / k) / (1 - 1 / k), probs

        return infer, {**PPLX, "device": args.device}

    import torch
    from huggingface_hub import HfApi, snapshot_download

    revision = HfApi().model_info(args.model_id, revision=args.revision).sha
    model_path = snapshot_download(repo_id=args.model_id, revision=revision)
    sys.path.insert(0, model_path)
    from joint_schema_model import encode_record, load_release_model, systemone

    model, processor = load_release_model(model_path, device=args.device, dtype=torch.bfloat16)

    def infer(request):
        clef.check_no_truncation(encode_record, processor, request, args.max_length)  # raises when too long
        return clef.parse_clef_response(systemone(model, processor, request, max_length=args.max_length))

    return infer, {"model_id": args.model_id, "resolved_revision": revision, "max_length": args.max_length}


def grade(infer, unit, model_id):
    """Grade one unit, leaving out the oldest exchanges while the judge rejects it as too long."""
    transcript = list(unit["transcript"])
    while True:
        try:
            choice, confidence, probs = infer(unit_request(unit, transcript, model_id))
            return {"choice": choice, "confidence": confidence, "probabilities": probs,
                    "dropped": len(unit["transcript"]) - len(transcript), "status": "ok"}
        except Exception as exc:  # over the judge's limit, or a bad unit
            if len(transcript) <= 1:
                return {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                        "dropped": len(unit["transcript"]) - len(transcript)}
            transcript = transcript[2:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--judge", choices=("pplx", "clef"), default="pplx")
    ap.add_argument("--units", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-id", default=clef.DEFAULT_MODEL_ID, help="clef only")
    ap.add_argument("--revision", default=clef.DEFAULT_MODEL_REVISION, help="clef only")
    ap.add_argument("--max-length", type=int, default=16384, help="clef only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, help="only the first N units (a smoke test; --run again resumes)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = ap.parse_args(argv)

    units_path = Path(args.units)
    units = [json.loads(line) for line in units_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    calls = plan(units)[: args.limit]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_id = args.model_id if args.judge == "clef" else PPLX["repo"]
    print(f"{len(units)} units -> {len(calls)} {args.judge} calls")
    if args.inspect_only:
        clef.write_json(out / "plan.json", {"units_file_sha256": clef.file_sha256(units_path), "calls": [
            {"case_id": u["case_id"], "request": unit_request(u, u["transcript"], model_id)} for u in calls]})
        print(f"Wrote {out / 'plan.json'}. No model was downloaded or called.")
        return 0

    started = time.perf_counter()
    infer, manifest = load_infer(args)
    clef.write_json(out / "manifest.json", {
        "judge": args.judge, "criteria": CRITERIA, **manifest,
        "load_seconds": time.perf_counter() - started, "units_file_sha256": clef.file_sha256(units_path),
        "runner_file_sha256": clef.file_sha256(Path(__file__)), "suite_commit": clef.git_head(ROOT),
    })
    results = out / "results.jsonl"
    done = set()
    if results.exists():
        done = {r["case_id"] for r in map(json.loads, results.read_text(encoding="utf-8").splitlines())}
    errors = 0
    with results.open("a", encoding="utf-8") as f:
        for i, unit in enumerate(calls, 1):
            if unit["case_id"] in done:
                continue
            started = time.perf_counter()
            row = {"case_id": unit["case_id"], **grade(infer, unit, model_id),
                   "elapsed_seconds": time.perf_counter() - started}
            errors += row["status"] != "ok"
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"{i}/{len(calls)} ({errors} errors)", flush=True)
    print(f"done: {len(calls)} calls, {errors} errors -> {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
