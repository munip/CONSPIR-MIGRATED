"""Run one model for Experiment 4. Each invocation is one model process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment import (
    EXPERIMENT_DIR,
    QUESTION_ID,
    SHUFFLE_SEED,
    json_sha256,
    laya_window_bounds,
    load_verified_train,
    make_inference_payload,
    run_documents,
    shuffled_documents,
)

ATTEMPTS_DIR = EXPERIMENT_DIR / "attempts"
PROTOCOL = json.loads((EXPERIMENT_DIR / "protocol.json").read_text(encoding="utf-8"))
WARMUP_TEXTS = (
    "Synthetic warm-up: a solver gives a direct answer and does not discuss its evaluation.",
    "Synthetic warm-up: the solver describes a rubric, then answers the stated question. This text exists only to initialize model kernels.",
    ("Synthetic warm-up paragraph about evidence, evaluation, and a solver's answer. " * 65).strip(),
    ("Synthetic warm-up paragraph about scoring rules and independent review. " * 190).strip(),
    ("Synthetic long warm-up paragraph about an answer, a rubric, and the need for independent grading. " * 330).strip(),
)


class DeviceFallbackError(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()


def new_attempt(model: str, shuffled: list[dict[str, Any]], evidence: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    ATTEMPTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_id = f"{timestamp}-{model}-{uuid.uuid4().hex[:8]}"
    path = ATTEMPTS_DIR / attempt_id
    path.mkdir(exist_ok=False)
    manifest = {
        "experiment": 4,
        "attempt_id": attempt_id,
        "model": model,
        "model_protocol": PROTOCOL["models"][model],
        "started_at_utc": now_utc(),
        "completion_status": "running",
        "phase": "initialization",
        "requested_documents": len(shuffled),
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_order_sha256": json_sha256([row["case_id"] for row in shuffled]),
        "dataset": evidence,
        "hardware": hardware_info(),
        "power_state": power_state(),
        "software": {"python": sys.version, "platform": platform.platform()},
        "warnings": [],
        "resource_samples": [],
    }
    write_json(path / "manifest.json", manifest)
    return path, manifest


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    return result.stdout.strip()


def hardware_info() -> dict[str, Any]:
    result = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        result["physical_memory_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        result["physical_memory_bytes"] = None
    return result


def power_state() -> dict[str, Any]:
    return {"captured_at_utc": now_utc(), "battery": command_output(["pmset", "-g", "batt"])}


def peak_rss_bytes() -> int | None:
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (AttributeError, OSError, ValueError):
        return None


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def snapshot_bytes(path: str | Path) -> int:
    return sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())


def fixed_warmup_rows(adapter: Any, synchronize: Any, output_path: Path) -> list[dict[str, Any]]:
    rows = []
    for index, text in enumerate(WARMUP_TEXTS):
        try:
            payload = make_inference_payload(text)
            synchronize()
            raw = adapter.infer(payload)
            synchronize()
            rows.append({
                "index": index,
                "synthetic_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "document_tokens": raw["document_tokens"],
                "window_count": raw["window_count"],
                "yes_probability": raw["answer"]["noul"] if "answer" in raw else raw["probabilities"]["true"],
            })
            write_json(output_path, {"requested": 5, "completed": len(rows), "inputs": rows})
        except BaseException as error:
            write_json(output_path, {"requested": 5, "completed": len(rows), "inputs": rows, "failure": {"type": type(error).__name__, "message": str(error)}})
            raise
    if len(rows) != 5:
        raise ValueError("expected five fixed warm-up inputs")
    return rows


def prepare_kev(manifest: dict[str, Any], documents: list[dict[str, Any]], warning_sink: list[str]) -> tuple[Any, Any, list[dict[str, Any]]]:
    import torch
    from huggingface_hub import snapshot_download
    from kev.api import SystemOneRequest, to_record
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

    model = PROTOCOL["models"]["kev"]
    download_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        checkpoint = Checkpoint(f"{model['repo']}@{model['revision']}")
        adapter_path = checkpoint.path
        base_path = snapshot_download(
            checkpoint.meta.base,
            revision=checkpoint.meta.base_revision,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja"],
        )
    warning_sink.extend(str(item.message) for item in captured)
    manifest["download_seconds"] = time.perf_counter() - download_started
    manifest["artifact_size_bytes"] = snapshot_bytes(adapter_path) + snapshot_bytes(base_path)
    manifest["artifact_bytes"] = {"kev_checkpoint": snapshot_bytes(adapter_path), "backbone": snapshot_bytes(base_path)}

    load_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        tokenizer, loaded_model = checkpoint.load(
            "mps", LoadOptions(dtype=torch.bfloat16, backend="mlx", temperature=None)
        )
    warning_sink.extend(str(item.message) for item in captured)
    manifest["load_seconds"] = time.perf_counter() - load_started
    if loaded_model.backend != "mlx" or loaded_model.dtype != "bfloat16":
        raise DeviceFallbackError(f"Kev backend/dtype mismatch: {loaded_model.backend}/{loaded_model.dtype}")
    manifest["effective_precision"] = {
        "backend": loaded_model.backend,
        "weights_dtype": loaded_model.dtype,
        "checkpoint_temperature": float(loaded_model.head.temperature),
    }
    manifest["software"].update(package_versions(("kev", "torch", "transformers", "mlx", "mlx-lm", "huggingface-hub")))
    manifest["backend_memory"] = {"after_load": kev_memory_counters()}

    def internal_record(payload: dict[str, Any]) -> dict[str, Any]:
        request = SystemOneRequest.model_validate(payload)
        record, _ = to_record(request)
        return record

    def encode(text: str) -> dict[str, Any]:
        return loaded_model.encode(
            tokenizer,
            internal_record(make_inference_payload(text)),
            max_state=SERVE_MAX_STATE,
            max_branch=SERVE_MAX_BRANCH,
            strict=True,
        )

    preflight = []
    for document in documents:
        encoded = encode(document["text"])
        if encoded["state_truncated"]:
            raise ValueError(f"Kev truncated {document['case_id']}")
        preflight.append({"case_id": document["case_id"], "document_tokens": encoded["state_tokens"] - 1, "complete": True})

    def infer(payload: dict[str, Any]) -> dict[str, Any]:
        encoded = loaded_model.encode(
            tokenizer,
            internal_record(payload),
            max_state=SERVE_MAX_STATE,
            max_branch=SERVE_MAX_BRANCH,
            strict=True,
        )
        logits = loaded_model.forward(encoded)[0]
        probabilities = torch.softmax(logits.float(), dim=-1).detach().cpu().tolist()
        p_false, p_true = float(probabilities[0]), float(probabilities[1])
        tokens = encoded["state_tokens"] - 1
        return {
            "probabilities": {"false": p_false, "true": p_true},
            "document_tokens": tokens,
            "model_input_tokens": len(encoded["ids"]),
            "window_count": 1,
            "windows": [{"index": 0, "token_start": 0, "token_end": tokens, "yes_probability": p_true}],
        }

    import mlx.core as mx

    def synchronize() -> None:
        mx.synchronize()
        if hasattr(torch, "mps"):
            torch.mps.synchronize()

    adapter = type("KevAdapter", (), {"infer": staticmethod(infer)})()
    return adapter, synchronize, preflight


def kev_memory_counters() -> dict[str, Any]:
    try:
        import mlx.core as mx
    except ImportError:
        return {}
    result = {}
    metal = getattr(mx, "metal", None)
    for key, name in (("active_bytes", "get_active_memory"), ("peak_bytes", "get_peak_memory"), ("cache_bytes", "get_cache_memory")):
        getter = getattr(mx, name, None)
        if getter is None and metal is not None:
            getter = getattr(metal, name, None)
        if getter:
            try:
                result[key] = int(getter())
            except Exception as error:
                result[key] = {"unavailable": str(error)}
    return result


def prepare_laya(manifest: dict[str, Any], documents: list[dict[str, Any]], warning_sink: list[str]) -> tuple[Any, Any, list[dict[str, Any]]]:
    import torch
    from huggingface_hub import snapshot_download
    from laya.agent import encode_text, load, serialize_state, window_budget

    model = PROTOCOL["models"]["laya"]
    download_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model_path = snapshot_download(
            model["repo"],
            revision=model["revision"],
            allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"],
        )
    warning_sink.extend(str(item.message) for item in captured)
    manifest["download_seconds"] = time.perf_counter() - download_started
    manifest["artifact_size_bytes"] = snapshot_bytes(model_path)
    manifest["artifact_bytes"] = {"english_checkpoint": manifest["artifact_size_bytes"]}

    load_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        agent = load(model["repo"], device="mps", revision=model["revision"])
    warning_sink.extend(str(item.message) for item in captured)
    manifest["load_seconds"] = time.perf_counter() - load_started
    if agent.device.type != "mps":
        raise DeviceFallbackError(f"Laya loaded on {agent.device}, not MPS")
    manifest["effective_precision"] = {
        "device": str(agent.device),
        "parameter_dtype": str(agent.dtype),
        "autocast_enabled": bool(agent.amp_enabled),
        "mps_amp_min_rows": int(agent.mps_amp_min_rows),
        "temperature_raw": agent.temperature_raw,
        "calibration_file": None,
    }
    manifest["software"].update(package_versions(("laya", "torch", "transformers", "numpy", "safetensors", "huggingface-hub")))
    manifest["backend_memory"] = {"after_load": laya_memory_counters(torch)}
    question = make_inference_payload("")["questions"]
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)

    def token_layout(text: str) -> tuple[int, list[tuple[int, int]]]:
        safe_text = serialize_state(text).replace(agent.tok.mask_token, " ")
        state_ids = encode_text(agent.tok, safe_text, add_special_tokens=False)["input_ids"]
        qid = QUESTION_ID
        for name in question:
            agent._check_question(name, question[name])
        internal = [agent._to_internal(question[qid])]
        budget, step, room = window_budget(agent.tok, internal, max_len, head_max_len)
        if len(state_ids) <= room:
            return len(state_ids), [(0, len(state_ids))]
        return len(state_ids), laya_window_bounds(len(state_ids), budget, step)

    preflight = []
    for document in documents:
        token_count, bounds = token_layout(document["text"])
        preflight.append({
            "case_id": document["case_id"],
            "document_tokens": token_count,
            "window_count": len(bounds),
            "windows": [{"token_start": start, "token_end": end} for start, end in bounds],
            "complete": bool(bounds and bounds[0][0] == 0 and bounds[-1][1] == token_count),
        })
        if not preflight[-1]["complete"]:
            raise ValueError(f"Laya token windows do not cover {document['case_id']}")

    def infer(payload: dict[str, Any]) -> dict[str, Any]:
        text = payload["state"]
        token_count, bounds = token_layout(text)
        capture: dict[str, Any] = {}

        def observe_windows(ctx: Any) -> None:
            capture["states"] = list(ctx.states)
            capture["results"] = list(ctx.results or [])

        fallback_before = int(getattr(agent, "cpu_fallback_count", 0))
        qmap = payload["questions"]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = agent.predict_long(text, qmap, batch_size=1, on_predict_end=observe_windows)
        new_warnings = [str(item.message) for item in captured]
        warning_sink.extend(new_warnings)
        if int(getattr(agent, "cpu_fallback_count", 0)) != fallback_before:
            raise DeviceFallbackError("Laya used its automatic CPU fallback during inference")
        if agent.device.type != "mps":
            raise DeviceFallbackError(f"Laya device changed to {agent.device}")
        states = capture.get("states", [])
        window_results = capture.get("results", [])
        if len(states) != len(bounds) or len(window_results) != len(bounds):
            raise ValueError("Laya observer window count differs from stock predict_long")
        windows = []
        for index, ((start, end), window_result) in enumerate(zip(bounds, window_results)):
            answer = window_result["answers"][QUESTION_ID]
            windows.append({
                "index": index,
                "token_start": start,
                "token_end": end,
                "yes_probability": float(answer["noul"]),
                "input_tokens": (window_result.get("usage") or {}).get("state_tokens"),
            })
        if int((result.get("usage") or {}).get("windows", -1)) != len(windows):
            raise ValueError("Laya aggregate window count differs from recorded window results")
        return {
            "answer": result["answers"][QUESTION_ID],
            "document_tokens": token_count,
            "model_input_tokens": (result.get("usage") or {}).get("state_tokens"),
            "window_count": len(windows),
            "windows": windows,
            "warnings": new_warnings,
        }

    def synchronize() -> None:
        torch.mps.synchronize()

    adapter = type("LayaAdapter", (), {"infer": staticmethod(infer)})()
    adapter.agent = agent
    return adapter, synchronize, preflight


def laya_memory_counters(torch: Any) -> dict[str, Any]:
    result = {}
    namespace = getattr(torch, "mps", None)
    for key, name in (("current_allocated_bytes", "current_allocated_memory"), ("driver_allocated_bytes", "driver_allocated_memory")):
        getter = getattr(namespace, name, None) if namespace is not None else None
        if getter:
            try:
                result[key] = int(getter())
            except Exception as error:
                result[key] = {"unavailable": str(error)}
    return result


def finalize(path: Path, manifest: dict[str, Any], status: str, phase: str, error: BaseException | None = None) -> None:
    manifest["completion_status"] = status
    manifest["phase"] = phase
    manifest["finished_at_utc"] = now_utc()
    manifest["wall_seconds"] = time.perf_counter() - manifest["_started_clock"]
    manifest["peak_rss_bytes"] = peak_rss_bytes()
    manifest["power_state_end"] = power_state()
    if error is not None:
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
    manifest.pop("_started_clock", None)
    write_json(path / "manifest.json", manifest)


def execute(model: str) -> int:
    documents, release = load_verified_train()
    shuffled = shuffled_documents(documents)
    path, manifest = new_attempt(model, shuffled, release["evidence"])
    manifest["_started_clock"] = time.perf_counter()
    write_json(path / "manifest.json", manifest)
    warning_sink: list[str] = []
    phase = "initialization"
    try:
        if model == "kev":
            adapter, synchronize, preflight = prepare_kev(manifest, shuffled, warning_sink)
        else:
            adapter, synchronize, preflight = prepare_laya(manifest, shuffled, warning_sink)
        phase = "warmup"
        manifest["preflight_count"] = len(preflight)
        manifest["warnings"] = warning_sink
        write_json(path / "preflight.json", preflight)
        manifest["phase"] = "warmup"
        write_json(path / "manifest.json", manifest)
        warmups = fixed_warmup_rows(adapter, synchronize, path / "warmup.json")
        if model == "laya" and not any(row["window_count"] > 1 for row in warmups):
            raise ValueError("Laya warm-up did not include a multi-window synthetic input")
        manifest["warmup_count"] = len(warmups)
        manifest["backend_memory"]["after_warmup"] = kev_memory_counters() if model == "kev" else laya_memory_counters(__import__("torch"))
        phase = "scoring"
        manifest["phase"] = "scoring"
        write_json(path / "manifest.json", manifest)

        results_path = path / "results.jsonl"
        results_path.touch()

        def save_result(row: dict[str, Any]) -> None:
            append_jsonl(results_path, row)

        rows, completion = run_documents(model, shuffled, adapter.infer, synchronize, save_result)
        manifest["scored_result_count"] = len(rows)
        manifest["warnings"] = warning_sink
        manifest["backend_memory"]["after_scoring"] = kev_memory_counters() if model == "kev" else laya_memory_counters(__import__("torch"))
        failed = sum(row["status"] == "inference_failed" for row in rows)
        if completion != "complete":
            status = completion
        elif failed:
            status = "completed_with_inference_failures"
        else:
            status = "complete"
        finalize(path, manifest, status, "complete" if status == "complete" else "scoring")
        print(json.dumps({"attempt_directory": str(path), "model": model, "status": status, "results": len(rows)}))
        return 0 if status == "complete" else 2
    except BaseException as error:
        manifest["warnings"] = warning_sink
        status = "incomplete_initialization_failure" if phase == "initialization" else f"incomplete_{phase}_failure"
        if isinstance(error, DeviceFallbackError):
            status = "incomplete_device_fallback"
        elif "memory" in str(error).lower() or "outofmemory" in type(error).__name__.lower():
            status = "incomplete_memory_failure"
        finalize(path, manifest, status, phase, error)
        print(json.dumps({"attempt_directory": str(path), "model": model, "status": status, "error": str(error)}), file=sys.stderr)
        return 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("kev", "laya"))
    args = parser.parse_args()
    raise SystemExit(execute(args.model))


if __name__ == "__main__":
    main()
