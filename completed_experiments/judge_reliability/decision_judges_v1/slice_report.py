#!/usr/bin/env python3
"""Each judge's agreement with the reference, split by how the reference label was made.

    python slice_report.py --reference validity_v1/labels/reference_v1.csv results/<judge>_validity_v1 ...

Slices (#364): u001–u020 and u076–u150 have two human labellers (plus adjudication); u021–u075 have one
human, powercoding1, with Qwen3-32B as the second labeller and every disagreement going to the human.
Three of the four judges are Qwen-based, so a judge doing much better on the one-human slice would
hint that agreement there partly tracks the Qwen second label. Descriptive only; score_validity.py is
the pre-registered analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_validity import kappa  # noqa: E402


def slice_of(unit_id: str) -> str:
    return "one human (u021–u075)" if 21 <= int(unit_id[1:]) <= 75 else "two humans (u001–u020, u076–u150)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    ref = {r["unit_id"]: r["label"] for r in csv.DictReader(args.reference.open(encoding="utf-8"))}
    rows, lines = [], ["| Judge | Variant | Slice | n | Accuracy | κ |", "|---|---|---|---|---|---|"]
    for d in args.dirs:
        judge = d.name.replace("_validity_v1", "")
        preds = [json.loads(line) for line in (d / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        for variant in ("no_context", "preceding_exchange"):
            for name in ("two humans (u001–u020, u076–u150)", "one human (u021–u075)"):
                pairs = [(ref[p["unit_id"]], p["choice"]) for p in preds
                         if p["variant"] == variant and slice_of(p["unit_id"]) == name and p["unit_id"] in ref]
                if not pairs:
                    continue
                gold, pred = [g for g, _ in pairs], [p for _, p in pairs]
                acc = sum(g == p for g, p in pairs) / len(pairs)
                k = kappa(gold, pred)
                rows.append({"judge": judge, "variant": variant, "slice": name, "n": len(pairs), "accuracy": acc, "kappa": k})
                lines.append(f"| {judge} | {variant} | {name} | {len(pairs)} | {acc:.3f} | {k:.3f} |")
    table = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
