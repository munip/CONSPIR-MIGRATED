"""Build a blind human-labelling sheet from real ConspiracyBench solver logs.

    python eval/judge_reliability/make_validity_sheet.py --out DIR --n 150 LOG [LOG ...]

Each unit is one model reply: the item's question, ground truth, checklists and notes, the
preceding exchange for a later turn, and the reply. Model names never appear in the sheet;
DIR/key.csv maps unit ids to (log, model, sample, epoch, turn) for scoring afterwards.
DIR/units.jsonl holds the same units in the eval/judge_cases.jsonl format (item, model_answer,
and for a later turn the transcript the Inspect scorer's judge sees), so any judge can be run
on them; expected_label stays empty until the human labels are adjudicated.

Sampling is stratified on features the judge doesn't decide, so the set isn't tilted towards
what any judge finds easy: framing (neutral / leading / adversarial) x turn (opening /
pressure) x source model. Adversarial and leading framings and pressure turns are
oversampled, because that's where diverge and abstain live (the #122 panel had 2 diverge and 0
abstain). Empty replies are skipped (they're invalid since #137). Replies cut off at max_tokens
are kept but flagged, since how to label them is part of what we measure.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

WEIGHT = {("adversarial", "pressure"): 3, ("adversarial", "opening"): 3, ("leading", "pressure"): 3,
          ("leading", "opening"): 2, ("neutral", "pressure"): 2, ("neutral", "opening"): 1}


def units_from_log(path):
    log = read_eval_log(path, resolve_attachments=True)
    model = log.eval.model
    for s in log.samples:
        item = s.metadata["item"]
        calls = [e for e in s.events if e.event == "model" and e.role is None and not e.error]
        history = []
        for turn, (e, m) in enumerate(zip(calls, [m for m in s.messages if m.role == "assistant"])):
            user_turns = [x for x in s.messages if x.role == "user"]
            prompt = user_turns[turn].text if turn < len(user_turns) else ""
            reply = m.text
            if reply.strip():
                yield {
                    "item": item, "turn": turn, "reply": reply, "prompt": prompt,
                    "history": list(history),
                    # What the scorer passes to build_judge_prompt for this turn: everything before
                    # the reply, including this turn's pressure message. Empty for the opening turn.
                    "transcript": history + [{"role": "user", "content": prompt}] if turn else [], "cut_off": e.output.stop_reason == "max_tokens",
                    "framing": item.get("framing", "neutral"), "kind": "opening" if turn == 0 else "pressure",
                    "source": {"log": Path(path).name, "model": model, "sample": str(s.id), "epoch": s.epoch, "turn": turn},
                }
            history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]


def sample(units, n, seed):
    rng = random.Random(seed)
    strata = defaultdict(list)
    for u in units:
        strata[(u["framing"], u["kind"], u["source"]["model"])].append(u)
    weights = {k: WEIGHT.get(k[:2], 1) for k in strata}
    total = sum(weights[k] * len(v) for k, v in strata.items())
    picked = []
    for k, v in sorted(strata.items()):
        take = min(len(v), round(n * weights[k] * len(v) / total))
        picked += rng.sample(v, take)
    rng.shuffle(picked)
    return picked[:n]


def history_text(history):
    return "\n\n".join(f"[{m['role'].upper()}] {m['content']}" for m in history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    units = [u for p in args.logs for u in units_from_log(p)]
    picked = sample(units, args.n, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sheet_cols = ["unit_id", "item_id", "framing", "turn", "cut_off", "question", "ground_truth_answer",
                  "diverges_if", "does_not_diverge_if", "notes", "preceding_exchange", "this_turn_user_message",
                  "model_reply", "label (hold/diverge/abstain)", "confidence (1-3)", "note"]
    with open(out / "sheet.csv", "w", newline="", encoding="utf-8") as f, \
         open(out / "key.csv", "w", newline="", encoding="utf-8") as k, \
         open(out / "units.jsonl", "w", encoding="utf-8") as uj:
        w, kw = csv.writer(f), csv.writer(k)
        w.writerow(sheet_cols)
        kw.writerow(["unit_id", "log", "model", "sample", "epoch", "turn"])
        for i, u in enumerate(picked, 1):
            it = u["item"]
            uid = f"u{i:03d}"
            w.writerow([uid, it.get("id"), u["framing"], u["turn"], "y" if u["cut_off"] else "n", it["question"],
                        it["ground_truth_answer"], "\n".join(it.get("diverges_if") or []),
                        "\n".join(it.get("does_not_diverge_if") or []), it.get("notes", ""),
                        history_text(u["history"]), u["prompt"] if u["turn"] else "", u["reply"], "", "", ""])
            kw.writerow([uid, *u["source"].values()])
            uj.write(json.dumps({"case_id": uid, "item": it, "model_answer": u["reply"],
                                 "transcript": u["transcript"], "turn": u["turn"], "framing": u["framing"],
                                 "cut_off": u["cut_off"], "expected_label": None}, ensure_ascii=False) + "\n")
    strata = defaultdict(int)
    for u in picked:
        strata[(u["source"]["model"], u["framing"], u["kind"])] += 1
    print(f"{len(units)} candidate units -> {len(picked)} sampled")
    for key, c in sorted(strata.items()):
        print(" ", key, c)


if __name__ == "__main__":
    main()
