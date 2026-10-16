#!/usr/bin/env python3
"""Speed and cost of each decision judge, from the files run_decision_validity.py and run_all.sh write.

    python speed_report.py --price-per-hour 1.99 results/clef_validity_v1 results/kev_validity_v1 ...

Latency is over timed successful calls only (warm-up calls are never recorded). Model load and
warm-up are reported per session, since a preempted run that resumes loads the model again.
GPU time counts load, warm-up and every recorded call; cost is GPU time at the hourly price.
Validity is scored separately, by score_validity.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return math.nan
    s = sorted(values)
    pos = (len(s) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def telemetry(path: Path) -> dict:
    """nvidia-smi samples, one per second; header lines repeat once per session and are skipped."""
    if not path.exists():
        return {}
    util, power, mem = [], [], []
    for row in csv.reader(path.read_text(encoding="utf-8").splitlines()):
        if len(row) < 6 or row[0].strip() == "timestamp":
            continue
        try:
            util.append(float(row[1])), mem.append(float(row[2])), power.append(float(row[3]))
        except ValueError:  # "[N/A]" readings
            continue
    if not power:
        return {}
    return {"samples": len(power), "mean_gpu_util_pct": sum(util) / len(util), "mean_power_w": sum(power) / len(power),
            "max_memory_used_mib": max(mem), "energy_wh": sum(power) / 3600}


def summarise(out: Path, price_per_hour: float) -> dict:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8")) if (out / "manifest.json").exists() else {}
    rows, sessions = read_jsonl(out / "results.jsonl"), read_jsonl(out / "sessions.jsonl")
    ok = [r for r in rows if r.get("status") == "ok"]
    lat = [r["elapsed_seconds"] for r in ok]
    tokens = [(r["input_tokens"], r["elapsed_seconds"]) for r in ok if r.get("input_tokens")]
    overhead = sum(s.get("load_seconds", 0) + s.get("warmup_seconds", 0) for s in sessions)
    gpu_seconds = overhead + sum(r["elapsed_seconds"] for r in rows)
    peak = [s[k] for s in sessions for k in ("peak_gpu_memory_bytes", "load_peak_gpu_memory_bytes") if s.get(k)]
    mean = sum(lat) / len(lat) if lat else math.nan
    return {
        "judge": manifest.get("judge", out.name), "dir": str(out),
        "calls": len(rows), "ok": len(ok), "errors": len(rows) - len(ok),
        "sessions": len(sessions), "load_seconds": [s.get("load_seconds") for s in sessions],
        "warmup_seconds": [s.get("warmup_seconds") for s in sessions],
        "latency_seconds": {"mean": mean, "p50": percentile(lat, 50), "p90": percentile(lat, 90),
                            "p99": percentile(lat, 99), "max": max(lat) if lat else math.nan},
        "calls_per_second": len(lat) / sum(lat) if lat else math.nan,
        "input_tokens_per_second": sum(t for t, _ in tokens) / sum(e for _, e in tokens) if tokens else None,
        "peak_gpu_memory_gb": max(peak) / 1e9 if peak else None,
        "gpu_seconds": gpu_seconds, "cost_usd": gpu_seconds / 3600 * price_per_hour,
        "cost_usd_per_1000_calls": mean * 1000 / 3600 * price_per_hour,
        "telemetry": telemetry(out / "gpu_telemetry.csv"),
        "runtime": sessions[-1].get("runtime") if sessions else None,
    }


def fmt(x, spec=".3f"):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def markdown(summaries: list[dict], price_per_hour: float) -> str:
    lines = [f"Latency over timed successful calls; cost at ${price_per_hour} per hour of the machine.", "",
             "| Judge | OK / calls | Load s | Mean s | p50 s | p90 s | p99 s | Calls/s | Tokens/s | Peak GB | Mean W | Wh | GPU min | Cost $ | $ / 1k calls |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        lat, tel = s["latency_seconds"], s["telemetry"]
        load = " + ".join(fmt(x, ".0f") for x in s["load_seconds"]) or "–"
        lines.append(f"| {s['judge']} | {s['ok']} / {s['calls']} | {load} | {fmt(lat['mean'])} | {fmt(lat['p50'])} | "
                     f"{fmt(lat['p90'])} | {fmt(lat['p99'])} | {fmt(s['calls_per_second'], '.2f')} | "
                     f"{fmt(s['input_tokens_per_second'], '.0f')} | {fmt(s['peak_gpu_memory_gb'], '.1f')} | "
                     f"{fmt(tel.get('mean_power_w'), '.0f')} | {fmt(tel.get('energy_wh'), '.1f')} | "
                     f"{fmt(s['gpu_seconds'] / 60, '.1f')} | {fmt(s['cost_usd'], '.2f')} | {fmt(s['cost_usd_per_1000_calls'], '.3f')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--price-per-hour", type=float, required=True)
    ap.add_argument("--out", type=Path, help="write speed_summary.json and speed_summary.md here (default: next to the first dir)")
    args = ap.parse_args()
    summaries = [summarise(d, args.price_per_hour) for d in args.dirs if d.exists()]
    out = args.out or args.dirs[0].parent / "decision_judges_v1_speed"
    out.mkdir(parents=True, exist_ok=True)
    (out / "speed_summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    table = markdown(summaries, args.price_per_hour)
    (out / "speed_summary.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
