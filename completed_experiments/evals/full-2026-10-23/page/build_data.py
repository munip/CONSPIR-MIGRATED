"""Collect the 2026-10-23 baseline results into data.json for the visualization.

Reads completed_experiments/evals/full-2026-10-23/outputs/<model>/{units.jsonl, pplx/results.jsonl, report/*.json}.
Transcripts are kept for a set of featured items only (the upload cap is ~1.8 MB), with answers
shortened; every item gets its per-model counts.
"""
import collections
import json
from pathlib import Path

RUN = Path("/Users/laurabocek/ConspiracyBench/.claude/worktrees/cb-eval-baseline-prep/completed_experiments/evals/full-2026-10-23/outputs")
MODELS = [("qwen3.5-4b", "Qwen3.5 4B", "baseline"), ("gemma-4-e4b", "Gemma 4 E4B", ""),
          ("granite-4.2-3b", "Granite 4.2 3B", ""), ("nemotron-3-nano-4b", "Nemotron 3 Nano 4B", ""),
          ("k2-horizon-3.7b", "K2 Horizon 3.7B", "")]
VARIANTS = ("multiturn", "multiturn_first", "single_turn")
ANSWER_CHARS = 520
CODE = {"hold": "H", "diverge": "D", "abstain": "A"}


def metric(m):
    if not m or m.get("rate") is None:
        return None
    lo, hi = m.get("ci95") or (None, None)
    clo, chi = m.get("ci95_clustered") or (None, None)
    # Six decimals: rounding to four and again for display moved some rates by 0.1 point.
    return {"r": round(m["rate"], 6), "lo": lo and round(lo, 6), "hi": hi and round(hi, 6),
            "n": m.get("count"), "d": m.get("total"),
            "clo": clo and round(clo, 6), "chi": chi and round(chi, 6)}


def wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    return round(mid - half, 6), round(mid + half, 6)


def endorsement(strings):
    """Lower = endorses conspiracy theories less, over each question's five tries.

    ``e5`` ever endorses (at least one try diverges), ``x5`` always endorses (all five do), ``er`` the
    share of all tries that diverge. Questions with an invalid try are left out, as for pass@5.
    """
    valid = [s for s in strings if s and all(c in "HDA" for c in s)]
    n = len(valid)
    ever, always, tries = sum("D" in s for s in valid), sum(set(s) == {"D"} for s in valid), sum(s.count("D") for s in valid)
    out = {}
    for name, k, d in (("e5", ever, n), ("x5", always, n), ("er", tries, 5 * n)):
        lo, hi = wilson(k, d) if name != "er" else (None, None)  # tries of one question aren't independent
        out[name] = {"r": round(k / d, 6) if d else None, "lo": lo, "hi": hi, "n": k, "d": d}
    return out


def short(text, n=ANSWER_CHARS):
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


items, per_model, metrics, by_turn, breakdowns, conf = {}, {}, {}, {}, {}, {}
units_by_model = {}
for key, name, _ in MODELS:
    units = [json.loads(l) for l in (RUN / key / "units.jsonl").open()]
    labels = {}
    for line in (RUN / key / "pplx" / "results.jsonl").open():
        r = json.loads(line)
        labels[r["case_id"]] = (r.get("choice") if r.get("status") == "ok" else None, (r.get("probabilities") or {}).get("diverge"))
    units_by_model[key] = (units, labels)

    # Headline metrics and breakdowns from the reports.
    metrics[key], breakdowns[key] = {}, {}
    for v in VARIANTS:
        s = json.loads((RUN / key / "report" / f"pplx_{v}.json").read_text())
        h, rep, conv = s["headline"], s["repeated_sampling"] or {}, s.get("repeated_conversations") or {}
        metrics[key][v] = {
            "div": metric(h["divergence_rate"]), "abs": metric(h["abstention_rate"]),
            "con": metric(h.get("concession_rate")), "ret": metric(h.get("retreat_rate")),
            "p5": metric(rep.get("pass_at_k")), "w5": metric(rep.get("worst_of_k")), "a5": metric(rep.get("all_k_agree")),
            "cp5": metric(conv.get("pass_at_k")), "cw5": metric(conv.get("worst_of_k")), "ca5": metric(conv.get("all_k_agree")),
            "inv": metric(s["invalid_turns"]["invalid"]),
        }
        breakdowns[key][v] = {
            field: {val: {"div": metric(g["divergence_rate"]), "abs": metric(g["abstention_rate"])}
                    for val, g in groups.items()}
            for field, groups in s["breakdowns"].items() if field in ("category", "band", "framing", "region")
        }

    # Per-turn label counts (multi-turn eval, every epoch) and the judge's confidence.
    turns = collections.defaultdict(collections.Counter)
    pdiv = collections.Counter()
    per_item = collections.defaultdict(lambda: {"mt": {}, "first": {}, "single": {}})
    for u in units:
        lab, p = labels.get(u["case_id"], (None, None)) if not u.get("skip") else (None, None)
        code = CODE.get(lab, "I")
        if p is not None:
            pdiv[min(int(p * 20), 19)] += 1
        item_id = u["item"]["id"]
        items.setdefault(item_id, u["item"])
        if u["variant"] == "multiturn":
            turns[u["turn"]][code] += 1
            per_item[item_id]["mt"].setdefault(u["epoch"], {})[u["turn"]] = code
        elif u["variant"] == "multiturn_first":
            per_item[item_id]["first"][u["epoch"]] = code
        else:
            per_item[item_id]["single"][u["epoch"]] = code
    by_turn[key] = {str(t): dict(c) for t, c in sorted(turns.items())}
    conf[key] = [pdiv[i] for i in range(20)]
    per_model[key] = per_item


def conversation(epoch_turns):
    seq = [epoch_turns[t] for t in sorted(epoch_turns)]
    if "I" in seq:
        return "I"
    if "D" in seq:
        return "D"
    return "H" if all(c == "H" for c in seq) else "A"


# Every item: metadata and, per model, compact label strings (one char per epoch).
item_rows = []
for item_id in sorted(items):
    it = items[item_id]
    diff = it.get("difficulty") or {}
    row = {"id": item_id, "q": short(it["question"], 240), "cat": it.get("category"), "band": diff.get("band"),
           "fr": it.get("framing"), "reg": it.get("region"), "lang": it.get("language") or "en",
           "mt": bool(it.get("pressure_turns")), "m": {}}
    for key, _, _ in MODELS:
        p = per_model[key][item_id]
        if row["mt"]:
            row["m"][key] = {
                "c": "".join(conversation(p["mt"][e]) for e in sorted(p["mt"])),           # 5 whole conversations
                "f": "".join(p["first"][e] for e in sorted(p["first"])),                    # 5 first questions alone
                "t": ["".join(p["mt"][e].get(t, "-") for t in sorted(p["mt"][e])) for e in sorted(p["mt"])],
            }
        else:
            row["m"][key] = {"s": "".join(p["single"][e] for e in sorted(p["single"]))}
    item_rows.append(row)

# Endorsement scores per model and eval (lower = endorses less): a multi-turn try is a whole
# conversation, which endorses if any turn diverges.
for key, _, _ in MODELS:
    metrics[key]["multiturn"].update(endorsement([r["m"][key]["c"] for r in item_rows if r["mt"]]))
    metrics[key]["multiturn_first"].update(endorsement([r["m"][key]["f"] for r in item_rows if r["mt"]]))
    metrics[key]["single_turn"].update(endorsement([r["m"][key]["s"] for r in item_rows if not r["mt"]]))


# Featured items: transcripts (epoch 1) for the items where the models disagree most.
def disagreement(row):
    if row["mt"]:
        outs = [sum(c == "D" for c in row["m"][k]["c"]) for k, _, _ in MODELS]
    else:
        outs = [sum(c == "D" for c in row["m"][k]["s"]) for k, _, _ in MODELS]
    return (sum(1 for o in outs if o) > 0, max(outs) - min(outs), sum(outs))


ranked = sorted(item_rows, key=disagreement, reverse=True)
featured = [r["id"] for r in ranked if r["mt"]][:46] + [r["id"] for r in ranked if not r["mt"]][:14]

transcripts = {}
for item_id in featured:
    it = items[item_id]
    entry = {"gt": short(it.get("ground_truth_answer"), 600), "pt": it.get("pressure_turns") or [], "m": {}}
    for key, _, _ in MODELS:
        units, labels = units_by_model[key]
        variant = "multiturn" if it.get("pressure_turns") else "single_turn"
        conv = sorted((u for u in units if u["variant"] == variant and u["item"]["id"] == item_id and u["epoch"] == 1),
                      key=lambda u: u["turn"])
        turns = []
        for u in conv:
            lab, p = labels.get(u["case_id"], (None, None)) if not u.get("skip") else (None, None)
            turns.append({"a": short(u["model_answer"]), "l": CODE.get(lab, "I"), "p": p and round(p, 2),
                          "skip": u.get("skip")})
        entry["m"][key] = turns
    transcripts[item_id] = entry

data = {
    "models": [{"key": k, "name": n, "note": note} for k, n, note in MODELS],
    "metrics": metrics, "byTurn": by_turn, "breakdowns": breakdowns, "conf": conf,
    "items": item_rows, "featured": featured, "transcripts": transcripts,
    "meta": {"items": len(item_rows), "multiturn": sum(r["mt"] for r in item_rows),
             "pressureTurns": sum(len(items[i].get("pressure_turns") or []) for i in items),
             "date": "2026-10-23", "judge": "pplx-decider v1.1 27B", "corpusCommit": "67b62b375"},
}
out = Path(__file__).parent / "data.json"
out.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
print(f"{out.stat().st_size / 1e6:.2f} MB; {len(item_rows)} items; {len(featured)} featured")
