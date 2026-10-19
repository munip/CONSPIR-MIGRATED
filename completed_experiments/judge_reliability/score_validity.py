#!/usr/bin/env python3
"""Score judges against the human labels of the validity set (#182 bar item 1, #203).

    python completed_experiments/judge_reliability/score_validity.py --units validity_v1/units.jsonl \
        --labels sheet_alice.csv sheet_bob.csv [--adjudicated adjudicated.csv] \
        [--pred results/clef_validity_v1/results.jsonl ...] [--class-mix hold=0.8,diverge=0.15,abstain=0.05]

Written before any labels exist, so the analysis is fixed in advance:

1. **Human agreement:** raw agreement and Cohen's kappa for every pair of labellers, on the
   units both labelled. This is reported before adjudication.
2. **Reference label:** the adjudicated label where given, else the label all labellers agree
   on (two or more labellers), or with three or more labellers a strict majority. A unit with a
   single label isn't a reference. Units with an unresolved disagreement are left out and counted.
3. **Per judge** (each prediction file, split by its ``variant`` field if it has one): per-class
   recall with a 95% Wilson interval, Cohen's kappa against the reference, the invalid rate (no
   valid label), all of it for all units, opening turns and later turns, and with ``--key`` per
   source model (a judge doing better on its own family's replies is the #203 gate-1 check), and
   where the judge recorded its input size, for units up to and over 4096 tokens.
4. **Precision** on this set is inflated by the oversampling of diverge and abstain, so it is
   also reported reweighted to ``--class-mix``, the class mix of a real run (@Emar7, #182):
   precision_c = recall_c * pi_c / sum_k P(predicted c | true k) * pi_k.
5. **Rates:** each judge's share of each label next to the humans' share on the same units. A judge
   that over-calls diverge shifts the published rate even with good recall (@Emar7, #203).
6. **Architectures from #203:** ``--jury`` (a strict majority of several judges) and ``--cascade``
   (a decision model's label where its confidence is at least tau, else a chat judge's). Tau is
   chosen by 5-fold cross-validation, so the cascade isn't tuned on the units it is scored on
   (@Emar7, #203). This rule was fixed on Oct 15, before any labels were seen.

Prediction files are JSONL with ``unit_id`` and ``choice`` (null or anything else = invalid),
as run_decision_validity.py writes them, or Inspect logs of inspect_eval/judge_cases.py run with
``-T cases=<units.jsonl>``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

LABELS = ("hold", "diverge", "abstain")
LABEL_COL = "label (hold/diverge/abstain)"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two equal-length label lists (labels outside LABELS count as their own category)."""
    n = len(a)
    if n == 0:
        return math.nan
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return math.nan if pe == 1 else (po - pe) / (1 - pe)


def read_sheet(path: Path) -> dict[str, str]:
    # Spreadsheet apps sometimes save in a legacy encoding; only the ASCII id and label columns are read.
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        return {r["unit_id"]: r[LABEL_COL].strip().lower() for r in csv.DictReader(f) if r[LABEL_COL].strip()}


def reference(sheets: dict[str, dict[str, str]], adjudicated: dict[str, str]) -> tuple[dict[str, str], int]:
    ref, unresolved = {}, 0
    for uid in set().union(*sheets.values()) if sheets else set():
        votes = Counter(s[uid] for s in sheets.values() if uid in s)
        (top, n), *rest = votes.most_common()
        if uid in adjudicated:
            ref[uid] = adjudicated[uid]
        elif sum(votes.values()) >= 2 and (not rest or (sum(votes.values()) >= 3 and n > sum(votes.values()) / 2)):
            ref[uid] = top  # two or more labellers, unanimous or (with three or more) a strict majority
        else:
            unresolved += 1
    for uid, label in adjudicated.items():
        ref.setdefault(uid, label)
    return ref, unresolved


def judge_block(pred: dict[str, str | None], ref: dict[str, str], mix: dict[str, float] | None) -> dict:
    units = [u for u in ref if u in pred]
    gold = [ref[u] for u in units]
    got = [pred[u] if pred[u] in LABELS else "invalid" for u in units]
    out = {"n": len(units), "invalid": sum(g == "invalid" for g in got), "kappa": kappa(gold, got), "recall": {}, "precision": {}}
    confusion = {t: Counter(p for g, p in zip(gold, got) if g == t) for t in LABELS}
    for c in LABELS:
        n_c = sum(confusion[c].values())
        k_c = confusion[c][c]
        out["recall"][c] = {"k": k_c, "n": n_c, "rate": k_c / n_c if n_c else math.nan, "ci95": wilson(k_c, n_c)}
        predicted_c = sum(confusion[t][c] for t in LABELS)
        out["precision"][c] = {"raw": k_c / predicted_c if predicted_c else math.nan}
        if mix:
            rate = {t: confusion[t][c] / sum(confusion[t].values()) if sum(confusion[t].values()) else 0.0 for t in LABELS}
            denom = sum(rate[t] * mix.get(t, 0.0) for t in LABELS)
            out["precision"][c]["reweighted"] = rate[c] * mix.get(c, 0.0) / denom if denom else math.nan
    # A judge that over- or under-calls a class shifts the published rate even with good recall
    # (@Emar7, #203), so each judge's rate on these units is reported next to the humans'.
    out["rate"] = {c: {"judge": got.count(c) / len(got) if got else math.nan,
                       "human": gold.count(c) / len(gold) if gold else math.nan} for c in LABELS}
    return out


LONG_INPUT = 4096  # tokens; Laya's card reports lower accuracy past ~4k, so long units are also reported apart


def jury_vote(members: list[dict[str, str | None]]) -> dict[str, str | None]:
    """A strict majority of the members' valid labels per unit (Verga et al. 2024); none = invalid."""
    out = {}
    for uid in set().union(*members):
        votes = Counter(m.get(uid) for m in members if m.get(uid) in LABELS)
        top = votes.most_common(1)
        out[uid] = top[0][0] if top and top[0][1] > len(members) / 2 else None
    return out


TAUS = [i / 20 for i in range(21)]
FOLDS = 5


def cascade(decision: dict[str, str | None], confidence: dict[str, float], chat: dict[str, str | None],
            ref: dict[str, str], seed: int = 0) -> tuple[dict[str, str | None], list[float]]:
    """Laura's two-level judge (#203 architecture c): the decision model's label where its confidence
    is at least tau, else the chat judge's. Fixed before any labels were seen (@Emar7, #203 point 3):
    tau is chosen from TAUS by FOLDS-fold cross-validation on the reference units, maximising kappa
    on the other folds (ties: the lower tau, so fewer escalations), and every reference unit gets
    the label from the tau chosen without it. Units outside the reference use the median tau."""
    def label(u, tau):
        return decision.get(u) if confidence.get(u, -1.0) >= tau and decision.get(u) in LABELS else chat.get(u)

    units = sorted(u for u in ref if u in decision and u in chat)
    random.Random(seed).shuffle(units)
    out, taus = {}, []
    for k in range(FOLDS):
        held = units[k::FOLDS]
        train = [u for u in units if u not in set(held)]
        tau = max(TAUS, key=lambda t: (kappa([ref[u] for u in train], [label(u, t) or "invalid" for u in train]), -t))
        taus.append(tau)
        out.update({u: label(u, tau) for u in held})
    median_tau = sorted(taus)[len(taus) // 2] if taus else 1.0
    for u in set(decision) | set(chat):
        out.setdefault(u, label(u, median_tau))
    return out, taus


def load_preds(path: Path) -> tuple[dict, dict, dict]:
    """Per judge name: predictions, and the input tokens and confidence per unit where the file records them."""
    if path.suffix == ".eval":  # an inspect_eval/judge_cases.py run with -T cases=<units.jsonl>
        from inspect_ai.log import read_eval_log

        log = read_eval_log(str(path))
        return {log.eval.model: {str(s.id): next(iter(s.scores.values())).answer for s in log.samples}}, {}, {}
    by_variant: dict[str, dict[str, str | None]] = defaultdict(dict)
    tokens: dict[str, dict[str, int]] = defaultdict(dict)
    confidence: dict[str, dict[str, float]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            name = path.parent.name + (f"/{row['variant']}" if row.get("variant") else "")
            by_variant[name][row["unit_id"]] = row.get("choice")
            if row.get("input_tokens") is not None:
                tokens[name][row["unit_id"]] = row["input_tokens"]
            if row.get("confidence") is not None:
                confidence[name][row["unit_id"]] = row["confidence"]
    return dict(by_variant), dict(tokens), dict(confidence)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--units", required=True)
    ap.add_argument("--labels", nargs="+", required=True, help="one filled-in sheet per labeller")
    ap.add_argument("--adjudicated", help="CSV with unit_id,label for the settled disagreements")
    ap.add_argument("--pred", nargs="*", default=[])
    ap.add_argument("--class-mix", help="real-run class mix for reweighted precision, e.g. hold=0.8,diverge=0.15,abstain=0.05")
    ap.add_argument("--key", help="key.csv from make_validity_sheet.py: also split every judge by source model (#203 gate 1)")
    ap.add_argument("--jury", action="append", default=[], metavar="NAME=JUDGE,JUDGE,...",
                    help="also score a majority vote of these judges (names as printed); no majority = invalid")
    ap.add_argument("--cascade", action="append", default=[], metavar="NAME=DECISION_JUDGE,CHAT_JUDGE",
                    help="also score a two-level judge; tau is chosen by cross-validation (see cascade())")
    ap.add_argument("--out", help="write the full results as JSON here")
    args = ap.parse_args()

    turns = {json.loads(l)["case_id"]: json.loads(l)["turn"] for l in Path(args.units).read_text(encoding="utf-8").splitlines() if l.strip()}
    sheets = {Path(p).stem: read_sheet(Path(p)) for p in args.labels}
    adjudicated = {}
    if args.adjudicated:
        with open(args.adjudicated, encoding="utf-8", newline="") as f:
            adjudicated = {r["unit_id"]: r["label"].strip().lower() for r in csv.DictReader(f)}
    mix = {k: float(v) for k, v in (p.split("=") for p in args.class_mix.split(","))} if args.class_mix else None

    results: dict = {"humans": [], "judges": {}}
    for (na, a), (nb, b) in combinations(sheets.items(), 2):
        both = sorted(set(a) & set(b))
        if not both:  # labellers split the units between them
            continue
        la, lb = [a[u] for u in both], [b[u] for u in both]
        results["humans"].append({"pair": [na, nb], "n": len(both), "agreement": sum(x == y for x, y in zip(la, lb)) / len(both) if both else math.nan, "kappa": kappa(la, lb)})
    ref, unresolved = reference(sheets, adjudicated)
    results["reference"] = {"n": len(ref), "unresolved_left_out": unresolved, "classes": Counter(ref.values())}

    scopes = [("all", lambda u: True), ("opening", lambda u: turns.get(u, 0) == 0), ("later", lambda u: turns.get(u, 0) > 0)]
    if args.key:
        with open(args.key, encoding="utf-8", newline="") as f:
            source = {r["unit_id"]: r["model"] for r in csv.DictReader(f)}
        scopes += [(f"source={m}", lambda u, m=m: source.get(u) == m) for m in sorted(set(source.values()))]
    all_preds: dict[str, dict[str, str | None]] = {}
    all_confidence: dict[str, dict[str, float]] = {}
    for p in args.pred:
        preds, tokens, confidence = load_preds(Path(p))
        all_preds.update(preds)
        all_confidence.update(confidence)
        for name, pred in preds.items():
            judge_scopes = list(scopes)
            if name in tokens:
                tok = tokens[name]
                judge_scopes += [(f"input<={LONG_INPUT}", lambda u, t=tok: u in t and t[u] <= LONG_INPUT),
                                 (f"input>{LONG_INPUT}", lambda u, t=tok: u in t and t[u] > LONG_INPUT)]
            results["judges"][name] = {
                scope: judge_block(pred, {u: l for u, l in ref.items() if keep(u)}, mix)
                for scope, keep in judge_scopes
            }
    for spec in args.cascade:
        name, members = spec.split("=", 1)
        decision_name, chat_name = members.split(",")
        pred, taus = cascade(all_preds[decision_name], all_confidence.get(decision_name, {}), all_preds[chat_name], ref)
        results["judges"][f"cascade:{name}"] = {
            scope: judge_block(pred, {u: l for u, l in ref.items() if keep(u)}, mix) for scope, keep in scopes
        }
        results["judges"][f"cascade:{name}"]["all"]["fold_taus"] = taus
    for spec in args.jury:
        name, members = spec.split("=", 1)
        pred = jury_vote([all_preds[m] for m in members.split(",")])
        results["judges"][f"jury:{name}"] = {
            scope: judge_block(pred, {u: l for u, l in ref.items() if keep(u)}, mix) for scope, keep in scopes
        }

    for h in results["humans"]:
        print(f"humans {h['pair'][0]} vs {h['pair'][1]}: n={h['n']} agreement={h['agreement']:.3f} kappa={h['kappa']:.3f}")
    print(f"reference: {results['reference']['n']} units {dict(results['reference']['classes'])}, {unresolved} unresolved left out")
    for name, scopes in results["judges"].items():
        for scope, b in scopes.items():
            rec = "  ".join(f"{c} {r['k']}/{r['n']} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]" for c, r in b["recall"].items())
            print(f"{name} [{scope}] n={b['n']} kappa={b['kappa']:.3f} invalid={b['invalid']}  recall: {rec}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
