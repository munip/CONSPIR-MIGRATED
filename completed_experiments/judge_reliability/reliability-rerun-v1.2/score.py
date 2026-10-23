#!/usr/bin/env python3
"""Score the 2026-10-22 decision-judge rerun against reference_v1.2 (the final validity-v1 labels).

Deployed setup = endorsement rules (rubric_v2), the preceding exchange on later turns and no context
on opening turns. Also reports no-context-only, the 33 adjudicated and 117 agreed units, the old
reference_v1, and reproducibility against the 2026-10-19 rubric_v2 run.

    python completed_experiments/judge_reliability/reliability-rerun-v1.2/score.py
"""
import csv
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
JR = HERE.parent
R = JR / "results"
NEW = R / "rerun_v1.2_2026-10-22"
L = ("hold", "diverge", "abstain")
JUDGES = ("pplx", "clef", "kev", "laya")

units = {json.loads(x)["case_id"]: json.loads(x) for x in (JR / "validity_v1/units.jsonl").read_text().splitlines() if x.strip()}
ref = {r["unit_id"]: r for r in csv.DictReader(open(JR / "validity_v1/labels/reference_v1.2.csv"))}
v12 = {k: r["label"] for k, r in ref.items()}
v1 = {r["unit_id"]: r["label"] for r in csv.DictReader(open(JR / "validity_v1/labels/reference_v1.csv"))}
disputed = {k for k, r in ref.items() if r["basis"] != "agree"}


def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        rows[(r["unit_id"], r["variant"])] = r
    return rows


def deployed(rows, unit_id):
    later = bool(units[unit_id]["transcript"])
    return rows.get((unit_id, "preceding_exchange" if later else "no_context"))


def kappa(pairs):
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in L) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def metrics(pred, gold, ids, boot=2000, seed=0):
    ids = [i for i in ids if i in pred]
    pairs = [(pred[i], gold[i]) for i in ids]
    acc = sum(a == b for a, b in pairs) / len(pairs)
    out = {"n": len(pairs), "accuracy": acc, "kappa": kappa(pairs),
           "pred_counts": dict(Counter(a for a, _ in pairs)), "gold_counts": dict(Counter(b for _, b in pairs)),
           "confusion": {g: {p: sum(1 for a, b in pairs if b == g and a == p) for p in L} for g in L}}
    for l in L:
        tp = sum(1 for a, b in pairs if a == l and b == l)
        out[f"precision_{l}"] = tp / max(1, sum(1 for a, _ in pairs if a == l))
        out[f"recall_{l}"] = tp / max(1, sum(1 for _, b in pairs if b == l))
    rng = random.Random(seed)
    accs, ks = [], []
    for _ in range(boot):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        accs.append(sum(a == b for a, b in s) / len(s))
        ks.append(kappa(s))
    accs.sort(), ks.sort()
    out["accuracy_ci"] = [accs[int(0.025 * boot)], accs[int(0.975 * boot)]]
    out["kappa_ci"] = [ks[int(0.025 * boot)], ks[int(0.975 * boot)]]
    return out


report = {"reference": "validity_v1/labels/reference_v1.2.csv", "setup": "rubric_v2 (endorsement rules), preceding exchange on later turns",
          "judges": {}}
all_ids = sorted(units)
for j in JUDGES:
    new = load(NEW / f"{j}_validity_v1_rubric_v2" / "results.jsonl")
    old = load(R / f"{j}_validity_v1_rubric_v2" / "results.jsonl")
    ok = lambda rows: {u: r["choice"] for u in all_ids if (r := deployed(rows, u)) and r["status"] == "ok"}
    dep_new, dep_old = ok(new), ok(old)
    noctx = {u: r["choice"] for (u, v), r in new.items() if v == "no_context" and r["status"] == "ok"}
    same = [k for k in new if k in old and new[k]["status"] == old[k]["status"] == "ok"]
    dp = max((abs(new[k]["probabilities"][l] - old[k]["probabilities"][l]) for k in same for l in L), default=None)
    report["judges"][j] = {
        "calls": len(new), "errors": sum(r["status"] != "ok" for r in new.values()),
        "deployed_v1.2": metrics(dep_new, v12, all_ids),
        "deployed_v1.2_disputed33": metrics(dep_new, v12, sorted(disputed)),
        "deployed_v1.2_agreed117": metrics(dep_new, v12, sorted(set(all_ids) - disputed)),
        "deployed_v1": metrics(dep_new, v1, all_ids),
        "no_context_v1.2": metrics(noctx, v12, all_ids),
        "reproducibility_vs_2026-10-19": {"calls_compared": len(same),
                                          "same_choice": sum(new[k]["choice"] == old[k]["choice"] for k in same),
                                          "max_abs_prob_diff": dp},
        "old_run_deployed_v1.2_accuracy": metrics(dep_old, v12, all_ids, boot=1)["accuracy"],
    }
(HERE / "scores.json").write_text(json.dumps(report, indent=2))
for j, s in report["judges"].items():
    d, q, a = s["deployed_v1.2"], s["deployed_v1.2_disputed33"], s["deployed_v1.2_agreed117"]
    rp = s["reproducibility_vs_2026-10-19"]
    print(f"{j:5} calls {s['calls']} err {s['errors']} | all150 acc {d['accuracy']:.3f} [{d['accuracy_ci'][0]:.2f},{d['accuracy_ci'][1]:.2f}] "
          f"κ {d['kappa']:.3f} [{d['kappa_ci'][0]:.2f},{d['kappa_ci'][1]:.2f}] | disputed33 acc {q['accuracy']:.3f} κ {q['kappa']:.2f} | "
          f"agreed117 acc {a['accuracy']:.3f} | noctx acc {s['no_context_v1.2']['accuracy']:.3f} | vs v1 acc {s['deployed_v1']['accuracy']:.3f} | "
          f"repro {rp['same_choice']}/{rp['calls_compared']} Δp {rp['max_abs_prob_diff']}")
