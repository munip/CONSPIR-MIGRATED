#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import speed_report as sr  # noqa: E402


class SpeedReportTest(unittest.TestCase):
    def test_a_resumed_run_counts_both_loads_and_only_successful_calls_in_latency(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "kev_validity_v1"
            out.mkdir()
            (out / "manifest.json").write_text(json.dumps({"judge": "kev"}))
            (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in (
                {"unit_id": "u1", "status": "ok", "elapsed_seconds": 1.0, "input_tokens": 100},
                {"unit_id": "u2", "status": "ok", "elapsed_seconds": 3.0, "input_tokens": 300},
                {"unit_id": "u3", "status": "error", "elapsed_seconds": 0.5})))
            (out / "sessions.jsonl").write_text("\n".join(json.dumps(s) for s in (
                {"load_seconds": 10, "warmup_seconds": 2, "peak_gpu_memory_bytes": 9e9},
                {"load_seconds": 8, "warmup_seconds": 2, "peak_gpu_memory_bytes": 8e9})))
            (out / "gpu_telemetry.csv").write_text(
                "timestamp, utilization.gpu [%], memory.used [MiB], power.draw [W], clocks.sm [MHz], temperature.gpu\n"
                "t, 50, 1000, 360, 1980, 40\n"
                "timestamp, utilization.gpu [%], memory.used [MiB], power.draw [W], clocks.sm [MHz], temperature.gpu\n"
                "t, 100, 2000, [N/A], 1980, 41\n"
                "t, 70, 3000, 720, 1980, 42\n")
            s = sr.summarise(out, price_per_hour=3600.0)  # $1 per GPU-second keeps the arithmetic visible
        self.assertEqual((s["calls"], s["ok"], s["errors"], s["sessions"]), (3, 2, 1, 2))
        self.assertEqual(s["latency_seconds"]["mean"], 2.0)
        self.assertEqual(s["latency_seconds"]["p50"], 2.0)
        self.assertEqual(s["input_tokens_per_second"], 100.0)
        self.assertEqual(s["gpu_seconds"], 10 + 2 + 8 + 2 + 1.0 + 3.0 + 0.5)
        self.assertEqual(s["cost_usd"], s["gpu_seconds"])
        self.assertEqual(s["peak_gpu_memory_gb"], 9.0)
        self.assertEqual(s["telemetry"]["samples"], 2)
        self.assertEqual(s["telemetry"]["mean_power_w"], 540.0)
        self.assertIn("| kev | 2 / 3 | 10 + 8 |", sr.markdown([s], 3600.0))


if __name__ == "__main__":
    unittest.main()
