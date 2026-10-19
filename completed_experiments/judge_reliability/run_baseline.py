#!/usr/bin/env python3
"""ConspiracyBench baseline judge reliability runner using the Groq direct provider.

Read-only with respect to the ConspiracyBench repository. The runner loads
regression cases, calls the repository's own build_judge_prompt(), and
uses Groq's direct API with openai/gpt-oss-20b.

Safety properties:
- --inspect-only makes zero model calls.
- --smoke-only makes exactly one model call.
- --run starts a fresh pass and refuses to overwrite prior partial results.
- --resume continues an interrupted pass without rerunning completed cases.
- Results are saved after every completed case.
- GROQ_API_KEY is read only from the environment and is never written to disk.
- No git commit/push/comment/merge operations are performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_LABELS = ("hold", "diverge", "abstain")
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_DELAY = 5.0
REQUEST_MAX_TOKENS = 512
REASONING_EFFORT = "low"
TEMPERATURE = 0.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_label(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip().lower().replace("`", "")
    aliases = {
        "holds": "hold", "held": "hold",
        "diverges": "diverge", "diverged": "diverge",
        "abstains": "abstain", "abstained": "abstain",
    }
    s = aliases.get(s, s)
    return s if s in VALID_LABELS else None


def strict_parse(text: str) -> str | None:
    s = text.strip().lower().replace("`", "")
    s = re.sub(r"^[\s\"']+|[\s\"'.!,;:]+$", "", s)
    return s if s in VALID_LABELS else None


# A negation anywhere earlier in the label's own clause (since the last . ! ? ; : or line break)
# denies it: "not", "n't", "no", "never", "neither", "nor", "cannot".
_NEGATION = r"\b(?:not|no|never|neither|nor|cannot)\b|n't\b"


def _affirmed(label: str, text: str) -> bool:
    """Whether ``label`` appears in ``text`` at least once with no negation earlier in its clause."""
    return any(
        not re.search(_NEGATION, re.split(r"[.!?;:\n]", text[: m.start()])[-1])
        for m in re.finditer(rf"\b{re.escape(label)}\b", text)
    )


def permissive_parse(text: str) -> str | None:
    strict = strict_parse(text)
    if strict:
        return strict
    tail = text.strip().lower()[-300:]
    # A label only counts where it isn't negated: "it does not diverge, it simply refuses"
    # names no verdict, and reading it as "diverge" would be a guess in the wrong direction.
    hits = [label for label in VALID_LABELS if _affirmed(label, tail)]
    if len(hits) == 1:
        return hits[0]
    m = re.search(
        r'["\']?(?:label|outcome|verdict)["\']?\s*[:=]\s*["\']?(hold|diverge|abstain)',
        tail,
    )
    return m.group(1) if m else None


def safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except ValueError:
        return f"external/{path.name}"


def snapshot_arcname(path: Path, repo: Path) -> str:
    try:
        rel = str(path.relative_to(repo)).replace("\\", "/")
        return f"snapshot/{rel}"
    except ValueError:
        return f"snapshot/external/{path.name}"


def find_case_file(repo: Path, custom_path: Path | None = None) -> Path:
    if custom_path is not None:
        cp = custom_path if custom_path.is_absolute() else (repo / custom_path)
        if not cp.exists():
            raise SystemExit(f"case file not found: {cp}")
        return cp
    for p in (repo / "eval" / "judge_cases.jsonl", repo / "evaluation" / "judge_cases.jsonl"):
        if p.exists():
            return p
    raise SystemExit("could not find eval/judge_cases.jsonl (or evaluation/judge_cases.jsonl).")


def find_prompt_file(repo: Path) -> Path:
    for p in (repo / "eval" / "judge_prompt.py", repo / "evaluation" / "judge_prompt.py"):
        if p.exists():
            return p
    raise SystemExit("could not find eval/judge_prompt.py (or evaluation/judge_prompt.py).")


def load_cases(path: Path) -> list[dict[str, Any]]:
    out = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSON on {path}:{lineno}: {exc}")
        if not isinstance(obj, dict):
            raise SystemExit(f"case on {path}:{lineno} is not a JSON object")
        out.append(obj)
    if not out:
        raise SystemExit(f"{path.name} is empty")
    return out


def validate_cases(cases: list[dict[str, Any]], is_default: bool = False) -> None:
    if is_default and len(cases) != 15:
        raise SystemExit(f"expected exactly 15 default regression cases, found {len(cases)}")
    seen_ids = set()
    for idx, c in enumerate(cases, 1):
        cid = c.get("case_id") or c.get("id")
        if cid:
            cid_str = str(cid)
            if cid_str in seen_ids:
                raise SystemExit(f"duplicate case_id found: {cid_str}")
            seen_ids.add(cid_str)
        exp = expected_label(c)
        if exp not in VALID_LABELS:
            raise SystemExit(f"case {cid or idx} has invalid or missing expected_label: {c.get('expected_label')}")


def detect_run_kind(cases: list[dict[str, Any]]) -> str:
    if any(c.get("transcript") for c in cases):
        return "multi_turn_regression"
    return "single_turn_regression"


def load_prompt_module(path: Path):
    spec = importlib.util.spec_from_file_location("cb_judge_prompt", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit("could not load judge_prompt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "build_judge_prompt", None)
    if not callable(fn):
        raise SystemExit("judge_prompt.py does not expose build_judge_prompt")
    return fn


def expected_label(case: dict[str, Any]) -> str | None:
    for k in ("expected_label", "expected", "label", "expected_outcome", "outcome"):
        v = norm_label(case.get(k))
        if v:
            return v
    return None


def answer_text(case: dict[str, Any]) -> str | None:
    for k in ("model_answer", "model_response", "response", "answer", "candidate_answer", "assistant_answer"):
        v = case.get(k)
        if isinstance(v, str):
            return v
    return None


def item_dict(case: dict[str, Any]) -> dict[str, Any]:
    nested = case.get("item")
    if isinstance(nested, dict):
        return dict(nested)
    exclude = {
        "expected_label", "expected", "label", "expected_outcome", "outcome",
        "case_type", "tags", "case_id", "name", "purpose",
    }
    ans_keys = {"model_answer", "model_response", "response", "answer", "candidate_answer", "assistant_answer"}
    return {k: v for k, v in case.items() if k not in exclude | ans_keys}


def build_prompt_adapter(fn, case: dict[str, Any]) -> str:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    ans = answer_text(case)
    item = item_dict(case)
    aliases: dict[str, Any] = {
        "case": case,
        "row": case,
        "example": case,
        "item": item,
        "question_item": item,
        "model_answer": ans,
        "model_response": ans,
        "response": ans,
        "answer": ans,
        "candidate_answer": ans,
        "question": item.get("question", case.get("question")),
        "ground_truth_answer": item.get("ground_truth_answer", case.get("ground_truth_answer")),
        "ground_truth_source": item.get("ground_truth_source", case.get("ground_truth_source")),
        "notes": item.get("notes", case.get("notes", "")),
        "diverges_if": item.get("diverges_if", case.get("diverges_if", [])),
        "does_not_diverge_if": item.get("does_not_diverge_if", case.get("does_not_diverge_if", [])),
        "transcript": case.get("transcript"),
    }
    args = []
    kwargs = {}
    missing = []
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name in aliases and aliases[p.name] is not None:
            val = aliases[p.name]
        elif p.name in case:
            val = case[p.name]
        elif p.default is not inspect._empty:
            continue
        else:
            missing.append(p.name)
            continue
        if p.kind == p.POSITIONAL_ONLY:
            args.append(val)
        else:
            kwargs[p.name] = val
    if missing:
        raise RuntimeError("Could not map build_judge_prompt parameter(s): " + ", ".join(missing))
    prompt = fn(*args, **kwargs)
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("build_judge_prompt did not return a non-empty string")
    return prompt


def case_id(case: dict[str, Any], idx: int) -> str:
    for k in ("case_id", "id", "name"):
        if case.get(k) is not None:
            return str(case[k])
    return f"case-{idx:03d}"


def detect_case_type(case: dict[str, Any]) -> str:
    for k in ("case_type", "type", "category"):
        if isinstance(case.get(k), str) and case[k].strip():
            return case[k].strip()
    tags = case.get("tags")
    if isinstance(tags, list) and tags:
        return ";".join(map(str, tags))
    return "unspecified"


def is_injection_case(case: dict[str, Any]) -> bool:
    hay = " ".join(str(case.get(k, "")) for k in ("case_type", "type", "tags", "name", "notes"))
    hay = (hay + " " + (answer_text(case) or "")).lower()
    needles = (
        "prompt injection", "injection", "ignore previous", "ignore the",
        "evaluator instruction", "judge instruction", "override",
    )
    return any(n in hay for n in needles)


def compute_metrics(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    labeled = [r for r in rows if r.get("expected_label") in VALID_LABELS]
    usable = [r for r in labeled if r.get(pred_key) in VALID_LABELS]
    correct = sum(r[pred_key] == r["expected_label"] for r in usable)
    confusion = {e: {p: 0 for p in VALID_LABELS} for e in VALID_LABELS}
    invalid = 0
    for r in labeled:
        pred = r.get(pred_key)
        if pred not in VALID_LABELS:
            invalid += 1
            continue
        confusion[r["expected_label"]][pred] += 1
    per_class = {}
    f1s = []
    for c in VALID_LABELS:
        tp = confusion[c][c]
        fp = sum(confusion[e][c] for e in VALID_LABELS if e != c)
        fn = sum(confusion[c][p] for p in VALID_LABELS if p != c)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion[c].values())}
        f1s.append(f1)
    return {
        "labeled_cases": len(labeled),
        "parsed_cases": len(usable),
        "invalid_output_count": invalid,
        "accuracy_over_parsed": correct / len(usable) if usable else None,
        "accuracy_counting_invalid_as_wrong": correct / len(labeled) if labeled else None,
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "per_class": per_class,
        "confusion": confusion,
    }


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def build_preflight_info(repo: Path, case_path: Path, prompt_path: Path, cases: list[dict[str, Any]], fn, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    labels = Counter(expected_label(c) or "MISSING" for c in cases)
    errors = []
    prompt_lengths = []
    nested_count = 0
    injection_count = 0
    for i, c in enumerate(cases, 1):
        if isinstance(c.get("item"), dict):
            nested_count += 1
        if is_injection_case(c):
            injection_count += 1
        try:
            p = build_prompt_adapter(fn, c)
            prompt_lengths.append(len(p))
        except Exception as exc:
            errors.append({"case_id": case_id(c, i), "error": repr(exc)})
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_kind": detect_run_kind(cases),
        "case_file": safe_relpath(case_path, repo),
        "prompt_file": safe_relpath(prompt_path, repo),
        "case_file_sha256": sha256_file(case_path),
        "prompt_file_sha256": sha256_file(prompt_path),
        "case_count": len(cases),
        "expected_label_counts": dict(labels),
        "nested_item_case_count": nested_count,
        "injection_case_count": injection_count,
        "prompt_build_errors": errors,
        "prompt_length_min": min(prompt_lengths) if prompt_lengths else None,
        "prompt_length_max": max(prompt_lengths) if prompt_lengths else None,
        "provider": "groq-direct",
        "model": model,
        "request_settings": {
            "reasoning_effort": REASONING_EFFORT,
            "include_reasoning": False,
            "max_completion_tokens": REQUEST_MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
    }


def preflight(repo: Path, case_path: Path, prompt_path: Path, cases: list[dict[str, Any]], fn, out_dir: Path, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    info = build_preflight_info(repo, case_path, prompt_path, cases, fn, model=model)
    write_json(out_dir / "baseline_preflight.json", info)
    return info


def ensure_groq_key() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in the environment before making API calls."
        )


def call_model(client, model: str, prompt: str) -> dict[str, Any]:
    t0 = time.time()
    result = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=REQUEST_MAX_TOKENS,
        temperature=TEMPERATURE,
        reasoning_effort=REASONING_EFFORT,
        include_reasoning=False,
        stream=False,
    )
    elapsed = time.time() - t0
    choice = result.choices[0]
    msg = choice.message
    content = getattr(msg, "content", None) or ""
    usage = getattr(result, "usage", None)

    def u(name: str):
        return getattr(usage, name, None) if usage is not None else None

    return {
        "content": str(content),
        "elapsed_seconds": elapsed,
        "finish_reason": getattr(choice, "finish_reason", None),
        "prompt_tokens": u("prompt_tokens"),
        "completion_tokens": u("completion_tokens"),
        "total_tokens": u("total_tokens"),
        "request_id": getattr(result, "id", None),
    }


def row_for_call(c: dict[str, Any], idx: int, call: dict[str, Any], model: str) -> dict[str, Any]:
    text = call["content"]
    exp = expected_label(c)
    strict = strict_parse(text)
    parsed = permissive_parse(text)
    return {
        "case_id": case_id(c, idx),
        "case_type": detect_case_type(c),
        "injection_case": is_injection_case(c),
        "expected_label": exp,
        "raw_output": text,
        "strict_label": strict,
        "parsed_label": parsed,
        "strict_correct": strict == exp if exp else None,
        "parsed_correct": parsed == exp if exp else None,
        "elapsed_seconds": call["elapsed_seconds"],
        "finish_reason": call["finish_reason"],
        "prompt_tokens": call["prompt_tokens"],
        "completion_tokens": call["completion_tokens"],
        "total_tokens": call["total_tokens"],
        "request_id": call["request_id"],
        "provider": "groq-direct",
        "model": model,
        "request_max_completion_tokens": REQUEST_MAX_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
        "include_reasoning": False,
        "temperature": TEMPERATURE,
    }


def finalize(out_dir: Path, repo: Path, case_path: Path, prompt_path: Path, rows: list[dict[str, Any]], model: str, run_kind: str) -> Path:
    strict_metrics = compute_metrics(rows, "strict_label")
    parsed_metrics = compute_metrics(rows, "parsed_label")
    inj_rows = [r for r in rows if r["injection_case"]]
    summary = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_kind": run_kind,
        "case_file": safe_relpath(case_path, repo),
        "prompt_file": safe_relpath(prompt_path, repo),
        "provider": "groq-direct",
        "model": model,
        "case_count": len(rows),
        "strict_metrics": strict_metrics,
        "permissive_metrics": parsed_metrics,
        "injection_subset_count": len(inj_rows),
        "injection_subset_accuracy": (
            sum(r["parsed_correct"] is True for r in inj_rows) / len(inj_rows) if inj_rows else None
        ),
        "request_settings": {
            "reasoning_effort": REASONING_EFFORT,
            "include_reasoning": False,
            "max_completion_tokens": REQUEST_MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
        "note": f"Judge reliability evaluation pass ({run_kind}, {len(rows)} cases) using the Groq direct API.",
    }
    write_json(out_dir / "baseline_summary.json", summary)

    with (out_dir / "baseline_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["expected\\predicted", *VALID_LABELS])
        conf = parsed_metrics["confusion"]
        for e in VALID_LABELS:
            w.writerow([e, *[conf[e][p] for p in VALID_LABELS]])

    fail_fields = ["case_id", "case_type", "injection_case", "expected_label", "parsed_label", "strict_label", "raw_output"]
    with (out_dir / "baseline_failures.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fail_fields)
        w.writeheader()
        for r in rows:
            if r["parsed_correct"] is not True or r["strict_label"] is None:
                w.writerow({k: r[k] for k in fail_fields})

    def fmt(v):
        return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    kind_display = run_kind.replace("_", " ").title()
    report = [
        f"# ConspiracyBench Judge Reliability — {kind_display}",
        "",
        f"- Run Kind: `{run_kind}`",
        f"- Fixture File: `{safe_relpath(case_path, repo)}`",
        f"- Provider: `Groq direct API`",
        f"- Model: `{model}`",
        f"- Cases: `{len(rows)}`",
        f"- Strict-format accuracy (invalid format counts wrong): `{fmt(strict_metrics['accuracy_counting_invalid_as_wrong'])}`",
        f"- Parsed-label accuracy (invalid output counts wrong): `{fmt(parsed_metrics['accuracy_counting_invalid_as_wrong'])}`",
        f"- Macro-F1 (parsed labels): `{fmt(parsed_metrics['macro_f1'])}`",
        f"- Invalid / unparseable outputs: `{parsed_metrics['invalid_output_count']}`",
        f"- Injection-like cases detected: `{len(inj_rows)}`",
        f"- Injection-subset parsed accuracy: `{fmt(summary['injection_subset_accuracy'])}`",
        "",
        "## Guardrail",
        "",
        "This is a small regression-set reliability check. It does not establish global judge validity. Review every failure before proposing benchmark changes.",
        "",
    ]
    (out_dir / "baseline_report.md").write_text("\n".join(report), encoding="utf-8")

    zip_path = out_dir / "conspiracybench_judge_reliability_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and p.resolve() != zip_path.resolve():
                z.write(p, arcname=str(p.relative_to(out_dir)))
        z.write(case_path, arcname=snapshot_arcname(case_path, repo))
        z.write(prompt_path, arcname=snapshot_arcname(prompt_path, repo))
    return zip_path


def make_partial_zip(out_dir: Path, repo: Path, case_path: Path, prompt_path: Path) -> Path:
    zpath = out_dir / "conspiracybench_judge_reliability_partial.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and p.resolve() != zpath.resolve():
                z.write(p, arcname=str(p.relative_to(out_dir)))
        z.write(case_path, arcname=snapshot_arcname(case_path, repo))
        z.write(prompt_path, arcname=snapshot_arcname(prompt_path, repo))
    return zpath


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Path to the ConspiracyBench repository root")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-only", action="store_true", help="No API calls; inspect assets only")
    mode.add_argument("--smoke-only", action="store_true", help="Exactly one Groq model call")
    mode.add_argument("--run", action="store_true", help="Start a new regression pass")
    mode.add_argument("--resume", action="store_true", help="Resume an interrupted Groq pass without rerunning completed cases")
    ap.add_argument("--case-file", type=Path, default=None, help="Path to alternate judge cases JSONL")
    ap.add_argument("--output-dir", type=Path, default=None, help="Directory for all outputs and bundles (defaults to repo/judge_reliability_results)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-calls", type=int, default=20)
    ap.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY)
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    if not repo.exists():
        raise SystemExit(f"repo path does not exist: {repo}")
    is_default = (args.case_file is None)
    case_path = find_case_file(repo, args.case_file)
    prompt_path = find_prompt_file(repo)
    cases = load_cases(case_path)
    validate_cases(cases, is_default=is_default)
    run_kind = detect_run_kind(cases)
    fn = load_prompt_module(prompt_path)

    if args.case_file is not None and args.output_dir is None:
        raise SystemExit(
            "--output-dir is required when --case-file is used "
            "so custom evaluations cannot overwrite baseline artifacts."
        )

    if args.output_dir is not None:
        out_dir = args.output_dir if args.output_dir.is_absolute() else (repo / args.output_dir)
    else:
        out_dir = repo / "judge_reliability_results"

    default_out_dir = (repo / 'judge_reliability_results').resolve()
    if args.case_file is not None and out_dir.resolve() == default_out_dir:
        raise SystemExit(
            'Custom --case-file evaluations must use an --output-dir '
            'different from the default judge_reliability_results directory.'
        )

    partial_path = out_dir / "baseline_results_partial.jsonl"
    progress_path = out_dir / "baseline_progress.json"
    completed: list[dict[str, Any]] = []

    # 1. Validate collision or resume provenance BEFORE mutating disk or checking credentials
    if args.run:
        if partial_path.exists() and partial_path.read_text(encoding="utf-8").strip():
            raise SystemExit(
                f"Partial Groq results already exist in {out_dir}. Use --resume to continue them, "
                "or specify a different --output-dir for a fresh run."
            )
    elif args.resume:
        if not partial_path.exists():
            raise SystemExit("no partial result file exists to resume.")
        if not progress_path.exists():
            raise SystemExit(
                f"Resume rejected: progress metadata file ({progress_path}) is missing. "
                "Cannot verify provenance of partial results."
            )
        try:
            prog = json.loads(progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise SystemExit(f"Resume rejected: corrupted progress file at {progress_path}")

        req_fields = ("case_file_sha256", "prompt_file_sha256", "model", "run_kind")
        for fld in req_fields:
            if not prog.get(fld):
                raise SystemExit(f"Resume rejected: missing required provenance field '{fld}' in {progress_path}.")

        expected_case_sha = sha256_file(case_path)
        expected_prompt_sha = sha256_file(prompt_path)

        if prog["case_file_sha256"] != expected_case_sha:
            raise SystemExit(
                f"Resume rejected: case file hash mismatch in {progress_path}. "
                "The partial results were generated from a different fixture."
            )
        if prog["prompt_file_sha256"] != expected_prompt_sha:
            raise SystemExit(f"Resume rejected: prompt file hash mismatch in {progress_path}.")
        if prog["model"] != args.model:
            raise SystemExit(f"Resume rejected: model mismatch (stored: {prog['model']}, requested: {args.model}).")
        if prog["run_kind"] != run_kind:
            raise SystemExit(f"Resume rejected: run_kind mismatch (stored: {prog['run_kind']}, requested: {run_kind}).")

        for raw in partial_path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                completed.append(json.loads(raw))

    if not args.inspect_only and not args.smoke_only and len(cases) > args.max_calls:
        raise SystemExit(f"{len(cases)} cases exceeds --max-calls={args.max_calls}.")

    # 2. Only write preflight metadata after all upfront checks pass
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = preflight(repo, case_path, prompt_path, cases, fn, out_dir, model=args.model)
    if pf["prompt_build_errors"]:
        raise SystemExit("Prompt-builder mapping failed; inspect the preflight output for details.")

    if args.inspect_only:
        zpath = out_dir / "judge_reliability_preflight.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out_dir.rglob("*")):
                if p.is_file() and p.resolve() != zpath.resolve():
                    z.write(p, arcname=str(p.relative_to(out_dir)))
            z.write(case_path, arcname=snapshot_arcname(case_path, repo))
            z.write(prompt_path, arcname=snapshot_arcname(prompt_path, repo))
        print("Preflight completed successfully. No API calls were made.")
        print("Created:", zpath)
        return 0

    # 3. Credentials and API clients required only when model execution is guaranteed
    ensure_groq_key()
    try:
        from groq import Groq, RateLimitError, APIStatusError, APITimeoutError, APIConnectionError
    except ImportError as exc:
        raise SystemExit("install dependency with: python -m pip install -U groq") from exc

    client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=0, timeout=60.0)

    if args.smoke_only:
        c = cases[0]
        prompt = build_prompt_adapter(fn, c)
        try:
            call = call_model(client, args.model, prompt)
        except Exception as exc:
            print(f"smoke call failed: {type(exc).__name__}: {exc}")
            return 3
        row = row_for_call(c, 1, call, args.model)
        write_json(out_dir / "baseline_smoke_result.json", row)
        print(json.dumps(row, indent=2, ensure_ascii=False))
        print("Smoke test complete: exactly 1 Groq model call was made.")
        return 0

    completed_ids = {r["case_id"] for r in completed}
    rows = list(completed)
    if completed_ids:
        print(f"Resuming with {len(completed_ids)} completed case(s): {', '.join(sorted(completed_ids))}")

    try:
        for i, c in enumerate(cases, 1):
            cid = case_id(c, i)
            if cid in completed_ids:
                continue
            prompt = build_prompt_adapter(fn, c)
            call = call_model(client, args.model, prompt)
            row = row_for_call(c, i, call, args.model)
            rows.append(row)
            completed_ids.add(cid)
            write_jsonl(partial_path, rows)
            write_json(progress_path, {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_kind": run_kind,
                "provider": "groq-direct",
                "model": args.model,
                "completed": len(rows),
                "total": len(cases),
                "completed_case_ids": [r["case_id"] for r in rows],
                "case_file_sha256": sha256_file(case_path),
                "prompt_file_sha256": sha256_file(prompt_path),
            })
            print(f"[{len(rows)}/{len(cases)}] {cid}: expected={row['expected_label']} parsed={row['parsed_label']} strict={row['strict_label']}")
            if len(rows) < len(cases) and args.delay_seconds > 0:
                time.sleep(args.delay_seconds)
    except (RateLimitError, APIStatusError, APITimeoutError, APIConnectionError) as exc:
        pzip = make_partial_zip(out_dir, repo, case_path, prompt_path)
        print(f"Groq API interrupted the run after {len(rows)}/{len(cases)} completed cases.")
        print(f"Error: {type(exc).__name__}: {exc}")
        print("Completed cases were saved and will not need to be rerun.")
        print("Partial bundle:", pzip)
        return 4
    except KeyboardInterrupt:
        pzip = make_partial_zip(out_dir, repo, case_path, prompt_path)
        print(f"\ninterrupted after {len(rows)}/{len(cases)} cases. Partial bundle: {pzip}")
        return 130

    if len(rows) != len(cases):
        pzip = make_partial_zip(out_dir, repo, case_path, prompt_path)
        print(f"run ended with {len(rows)}/{len(cases)} cases. Partial bundle: {pzip}")
        return 5

    # Stable case order before final artifacts.
    order = {case_id(c, i): i for i, c in enumerate(cases, 1)}
    rows.sort(key=lambda r: order[r["case_id"]])
    write_jsonl(out_dir / "baseline_results.jsonl", rows)
    zip_path = finalize(out_dir, repo, case_path, prompt_path, rows, args.model, run_kind)
    print("\nCreated:", zip_path)
    print(f"Judge reliability evaluation completed successfully: {run_kind}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
