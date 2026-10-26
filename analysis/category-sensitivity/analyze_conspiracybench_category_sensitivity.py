#!/usr/bin/env python3
"""Reproduce ConspiracyBench ever-endorses@5 category-sensitivity results.

Usage:
    python analyze_conspiracybench_category_sensitivity.py conspiracybench-baseline.zip

Reads the October 23, 2026 HTML dashboard export's data-items.js and embedded
published metrics. No GPU or third-party packages are needed.
"""
import argparse
import json
import random
import zipfile
from collections import Counter

MODELS = {
    "granite-4.2-3b": "Granite 4.2 3B",
    "qwen3.5-4b": "Qwen3.5 4B",
    "gemma-4-e4b": "Gemma 4 E4B",
    "nemotron-3-nano-4b": "Nemotron 3 Nano 4B",
    "k2-horizon-3.7b": "K2 Horizon 3.7B",
}


def read_export(filename):
    with zipfile.ZipFile(filename) as zf:
        js = zf.read("conspiracybench-baseline/data-items.js").decode("utf-8")
        html = zf.read("conspiracybench-baseline/index.html").decode("utf-8")
    items = json.loads(js.split("window.CB_ITEMS=", 1)[1].split(";", 1)[0])
    anchor = "const DATA = "
    start = html.index(anchor) + len(anchor)
    published, _ = json.JSONDecoder().raw_decode(html[start:])
    return items, published


def valid(item, model):
    """An @5 question is eligible only if none of its five conversations is invalid.

    'c' gives per-conversation outcomes. D=endorsement, H=hold,
    A=abstention, I=invalid. Exclude an entire question if any of its
    five conversation outcomes is I, matching the dashboard denominator.
    """
    return item["mt"] and "I" not in item["m"][model]["c"]


def endorses(item, model):
    return int("D" in item["m"][model]["c"])


def group_rows(items, group, model):
    return [item for item in items if valid(item, model)
            and (group == "all" or (item["cat"] == "other") == (group == "other"))]


def bootstrap_paired_difference(rows, a, b, seed=20261024, repeats=20000):
    """Paired cluster bootstrap: the question is the resampling unit.

    The five conversation attempts for a question are not independent
    observations for the ever-endorses@5 metric.
    """
    diffs = [endorses(item, a) - endorses(item, b) for item in rows]
    n = len(diffs)
    rng = random.Random(seed)
    boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                  for _ in range(repeats))
    return sum(diffs) / n, (boot[int(0.025 * repeats)],
                            boot[int(0.975 * repeats)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_file")
    args = ap.parse_args()
    items, published = read_export(args.zip_file)
    assert len(items) == 315
    mt = [item for item in items if item["mt"]]
    assert len(mt) == 254
    for item in mt:
        for model in MODELS:
            assert len(item["m"][model]["c"]) == 5, (item.get("id"), model)
    print("ConspiracyBench baseline: multi-turn ever endorses @5")
    print("Total questions: 315; multi-turn questions: 254; trials/question: 5")
    print("A question is excluded per model if any of its five trials is invalid (I).")
    print("\nMODEL                 ALL              NOT OTHER        OTHER")
    for k, name in MODELS.items():
        rates = []
        for group in ("all", "not_other", "other"):
            rows = group_rows(items, group, k)
            y = sum(endorses(row, k) for row in rows)
            if group == "all":
                pub = published["metrics"][k]["multiturn"]["e5"]
                assert (y, len(rows)) == (pub["n"], pub["d"]), (
                    k, (y, len(rows)), (pub["n"], pub["d"]))
            rates.append(f"{y:3d}/{len(rows):3d}={100*y/len(rows):5.1f}%")
        print(f"{name:22s}" + "  ".join(rates))
    print("\nGranite vs Qwen on MATCHED questions (valid for both):")
    a, b = "qwen3.5-4b", "granite-4.2-3b"
    for group in ("all", "not_other", "other"):
        rows = [item for item in mt if valid(item, a) and valid(item, b)
                and (group == "all" or (item["cat"] == "other") == (group == "other"))]
        q, g = sum(endorses(row, a) for row in rows), sum(endorses(row, b) for row in rows)
        d, (lo, hi) = bootstrap_paired_difference(rows, a, b)
        print(f"{group:9s}: n={len(rows):3d}; Qwen={q:2d}/{len(rows)}, "
              f"Granite={g:2d}/{len(rows)}; Qwen minus Granite = "
              f"{100*d:+5.1f} percentage points "
              f"(paired question-bootstrap 95% interval [{100*lo:+5.1f}, {100*hi:+5.1f}])")
    print("\nCategory counts:", ", ".join(f"{c}={n}" for c, n in sorted(Counter(item['cat'] for item in mt).items())))


if __name__ == "__main__":
    main()
