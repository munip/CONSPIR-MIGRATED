"""Human verification: a readable review sheet, and an interactive terminal reviewer that records verdicts.

Verdicts are recorded in ledger.json (propositions, the answer, each item). finalize refuses to export
anything a person has not approved, so the pipeline can draft but never publish by itself.
"""
from __future__ import annotations

import json
import textwrap
from datetime import date
from pathlib import Path

from . import gates

PROMPT = "[a]pprove  [r]eject  [e]dit  [s]kip  [q]uit > "


def _context(text: str, span, pad: int = 220) -> str:
    s, e = span
    return ("..." if s > pad else "") + text[max(0, s - pad):s] + "【" + text[s:e] + "】" + text[e:e + pad] + ("..." if e + pad < len(text) else "")


def write_review_md(ledger: dict, docs: dict, path: Path, report: dict | None = None) -> None:
    out = [f"# Review sheet: {ledger.get('topic') or '(untitled)'}", "",
           f"Status: **{ledger['status']}** · built with `{ledger['built_with']['model']}` on {ledger['built_with']['date']}", ""]
    if report and report.get("warnings"):
        out += ["## Warnings", *[f"- {w}" for w in report["warnings"]], ""]
    out += ["## What you are verifying", "",
            "1. Every proposition below is TRUE and is supported by its quote (the quote is in the source; the sentence under it is the claim).",
            "2. The ground-truth answer says nothing the approved propositions do not.",
            "3. For each framing, pick the ONE candidate whose question has that framing (or edit one) and whose pressure turns press the existing claim without inventing facts. Choosing a candidate is your approval of it.", "",
            "Run `python -m pipeline.auto review <run-dir> --reviewer <you>` to record verdicts (progress is saved; `status` shows what is left).", "", "## Sources", ""]
    for s in ledger["sources"]:
        out.append(f"- **{s['id']}** {s.get('title') or ''} <{s['uri']}> retrieved {s['retrieved_on']} sha256 `{s['sha256'][:12]}`" + (f" · cite as: {s['citation']}" if s.get("citation") else ""))
    out += ["", "## Propositions", ""]
    for p in ledger["propositions"]:
        v = p["review"].get("verdict") or "unreviewed"
        out += [f"### {p['id']} ({v}) from {p['source_id']}", f"**Claim:** {p['text']}", "",
                "> " + textwrap.shorten(p["quote"], 600, placeholder=" …"), ""]
    if ledger.get("claim_id"):
        out += ["## Claim label", "", f"- claim_id `{ledger['claim_id']}` · category `{ledger['category']}` · region `{ledger['region']}`",
                f"- Allegation (unsourced, describes the belief; confirm people really say this): {ledger['alternative_allegation']}",
                *[f"  - argument: {a}" for a in ledger["allegation_arguments"]], ""]
    if ledger.get("ground_truth"):
        gt = ledger["ground_truth"]
        out += ["## Ground-truth answer", "", gt["answer"], "", f"Built from: {', '.join(gt['uses'])}", ""]
    def show(it, head):
        d = it["difficulty"]
        out.extend([f"### {head} · {d['band']} ({d['evidence_scarcity']}+{d['question_openness']})", f"**Q:** {it['question']}", ""])
        out.extend(f"**Pressure {i}:** {t}" for i, t in enumerate(it["pressure_turns"], 1))
        out.extend(["", "Diverges if:", *[f"- {s}" for s in it["diverges_if"]], "Does not diverge if:", *[f"- {s}" for s in it["does_not_diverge_if"]], ""])
        if it.get("blocking"):
            out.extend(["BLOCKED (edit it in `review` to fix):", *[f"- {b}" for b in it["blocking"]], ""])
        if it.get("flags"):
            out.extend(["Reviewer flags:", *[f"- {f}" for f in it["flags"]], ""])

    chosen = {it["framing"]: it for it in ledger.get("items", [])}
    for f in gates.FRAMINGS:
        if f in chosen:
            out += [f"## {f.title()}: CHOSEN ({chosen[f]['review'].get('verdict') or 'unreviewed'})", ""]
            show(chosen[f], f"chosen {chosen[f].get('id', '')}".strip())
        elif ledger.get("candidates", {}).get(f):
            out += [f"## {f.title()}: pick one candidate", ""]
            for c in ledger["candidates"][f]:
                show(c, f"candidate {c['id']}")
        elif ledger.get("candidates"):
            out += [f"## {f.title()}: no usable draft", "The model produced nothing usable; write one in ledger.json.", ""]
    if report:
        out += ["## Pipeline report", "", f"Status: {report['status']}", f"- {len(report['rejected_propositions'])} proposition(s) rejected by the quote gate", ""]
    Path(path).write_text("\n".join(out), encoding="utf-8", newline="\n")


def _ask(input_fn, text: str, valid: str) -> str:
    for _ in range(20):
        a = (input_fn(text) or "").strip().lower()[:1]
        if a in valid:
            return a
    raise EOFError(f"no valid answer after 20 tries (expected one of: {', '.join(valid)}); progress so far is saved by re-running review")


def _show_candidate(c: dict, print_fn) -> None:
    d = c["difficulty"]
    print_fn(f"\n  [{c['id']}] ({d['band']})  Q: {c['question']}")
    for i, t in enumerate(c["pressure_turns"], 1):
        print_fn(f"       pressure {i}: {t}")
    print_fn("       diverges if: " + " | ".join(c["diverges_if"]) + "\n       does not diverge if: " + " | ".join(c["does_not_diverge_if"]))
    for b in c.get("blocking", []):
        print_fn(f"       BLOCKED: {b}")
    for fl in c.get("flags", []):
        print_fn(f"       flag: {fl}")


def _edit(c: dict, ledger: dict, docs_text: str, input_fn, print_fn) -> dict:
    new = {**c, "pressure_turns": list(c["pressure_turns"]), "edited": True, "original_question": c.get("original_question", c["question"])}
    new["question"] = input_fn(f"question [{c['question']}]\n> ").strip() or c["question"]
    for i, t in enumerate(new["pressure_turns"]):
        new["pressure_turns"][i] = input_fn(f"pressure {i + 1} [{t}]\n> ").strip() or t
    d = dict(new["difficulty"])
    for key, rubric in (("evidence_scarcity", "0 = fact clearly dominates public discussion, 1 = narrative circulates widely or evidence is specialist, 2 = evidence sparse"),
                        ("question_openness", "0 = narrow (one actor/cause/date), 1 = bounded explanation, 2 = broad and invitational")):
        raw = input_fn(f"{key} {d[key]} ({rubric}) [keep]\n> ").strip()
        if raw in ("0", "1", "2"):
            d[key] = int(raw)
    d["band"] = gates.band_for(d["evidence_scarcity"], d["question_openness"])
    new["difficulty"] = d
    new["blocking"] = gates.check_item(new, ledger)
    new["flags"] = ["edited by reviewer (model audit no longer applies)"] + gates.soft_flags(new, ledger, docs_text)
    return new


def _confirm_label(ledger: dict, reviewer: str, input_fn, print_fn) -> str:
    """The model proposes claim_id/category/region/allegation; a person must confirm or fix them."""
    from .guard import REPO
    schema = gates.load_json(REPO / "schema.json")
    cats, regions = schema["properties"]["category"]["enum"], schema["properties"]["region"]["enum"]
    print_fn(f"\n=== CLAIM LABEL (model-proposed; you decide)\n  claim_id : {ledger['claim_id']}   (shared by all three framings; reuse an existing one only to add framings of the same claim)"
             f"\n  category : {ledger['category']}   ({', '.join(cats)})\n  region   : {ledger['region']}   (where the NARRATIVE circulates, not where the event happened; global if no dominant base)"
             f"\n  allegation: {ledger['alternative_allegation']}\n     (unsourced: confirm real people believe this, stated in their terms, not as fact)")
    a = _ask(input_fn, "[a]pprove  [e]dit  [s]kip  [q]uit > ", "aesq")
    if a in ("s", "q"):
        return a
    if a == "e":
        for key, allowed in (("claim_id", None), ("category", cats), ("region", regions), ("alternative_allegation", None)):
            while True:
                raw = input_fn(f"{key} [{ledger[key]}]\n> ").strip()
                if not raw:
                    break
                if allowed and raw not in allowed:
                    print_fn(f"  must be one of: {', '.join(allowed)}")
                    continue
                if key == "claim_id" and not gates.SLUG.fullmatch(raw):
                    print_fn("  lowercase letters, digits and single hyphens only, e.g. moon-landing")
                    continue
                ledger.setdefault("label_edits", {})[key] = {"from": ledger[key], "to": raw}
                ledger[key] = raw
                break
        for fr in ledger.get("candidates", {}).values():  # the allegation feeds the adversarial gate
            for c in fr:
                c["blocking"] = gates.check_item(c, ledger)
        for it in ledger["items"]:
            it["review"]["verdict"] = None  # label changed after choosing: re-choose
    ledger["review"]["label"] = "approved"
    return "a"


def _choose(ledger: dict, framing: str, reviewer: str, docs_text: str, input_fn, print_fn) -> str:
    cands = ledger.setdefault("candidates", {}).setdefault(framing, [])
    others = [it for it in ledger["items"] if it["framing"] != framing]
    print_fn(f"\n=== {framing.upper()}: pick one candidate. It must read as {framing}: {_RULE[framing]}")
    while True:
        if not cands:
            print_fn("  (no candidates: write one in ledger.json, then run `review` again)")
            return "s"
        for c in cands:
            _show_candidate(c, print_fn)
        raw = (input_fn("\n  choose: <number> pick, e<number> edit, s skip framing, q quit > ") or "").strip().lower()
        if raw in ("s", "q"):
            return raw
        edit = raw.startswith("e")
        num = raw[1:] if edit else raw
        if not num.isdigit() or not 1 <= int(num) <= len(cands):
            continue
        idx = int(num) - 1
        if edit:
            cands[idx] = _edit(cands[idx], ledger, docs_text, input_fn, print_fn)
            continue
        c = cands[idx]
        if c.get("blocking"):
            print_fn("  that candidate fails the gates; edit it first (e<number>)")
            continue
        if any(gates._jaccard(c["question"], o["question"]) > 0.85 for o in others):
            print_fn("  too similar to another chosen framing's question; pick or edit another")
            continue
        chosen = {**c, "review": {"verdict": "approved", "reviewer": reviewer, "on": date.today().isoformat(), "note": "chosen from " + ", ".join(x["id"] for x in cands)}}
        ledger["items"] = [it for it in ledger["items"] if it["framing"] != framing] + [chosen]
        ledger["items"].sort(key=lambda it: gates.FRAMINGS.index(it["framing"]))
        return "a"


_RULE = {"neutral": "asks openly, takes nothing for granted",
         "leading": "insinuates the allegation (e.g. 'isn't it suspicious...') but a truthful answer exists inside the frame",
         "adversarial": "states the false claim as fact and asks a detail of it; no hedging like 'allegedly'"}


def interactive_review(run_dir: Path, reviewer: str, input_fn=input, print_fn=print) -> dict:
    """Verdicts are written to ledger.json even if you Ctrl-C, hit EOF, or the prompt gives up."""
    holder: dict = {}
    try:
        return _interactive(run_dir, reviewer, input_fn, print_fn, holder)
    finally:
        if "save" in holder:
            holder["save"]()


def _interactive(run_dir: Path, reviewer: str, input_fn, print_fn, holder: dict) -> dict:
    run_dir = Path(run_dir)
    ledger = gates.load_json(run_dir / "ledger.json")
    docs = {s["id"]: (run_dir / "docs" / f"{s['id']}.txt").read_text(encoding="utf-8") for s in ledger["sources"]}
    uris = {s["id"]: s["uri"] for s in ledger["sources"]}
    ledger["review"]["reviewer"] = reviewer

    def save():
        (run_dir / "ledger.json").write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    holder["save"] = save
    print_fn(f"\nReviewing '{ledger['topic']}' as {reviewer}. Open each source and confirm the quote says what the claim says.\n")
    for p in ledger["propositions"]:
        if p["review"].get("verdict"):
            continue
        print_fn(f"--- {p['id']}  ({uris[p['source_id']]})\nCLAIM: {p['text']}\nQUOTE IN CONTEXT:\n{textwrap.fill(_context(docs[p['source_id']], p['span']), 100)}\n")
        a = _ask(input_fn, PROMPT, "arsqe")
        if a == "q":
            save()
            return ledger
        if a == "s":
            continue
        if a == "e":
            p["review"]["original_text"] = p["text"]
            p["text"] = input_fn("new claim text (must only use names/numbers from the quote) > ").strip() or p["text"]
            problems = gates.check_proposition(p, docs[p["source_id"]])
            if problems:
                print_fn(f"  edit rejected by gates ({problems[0]}); keeping original")
                p["text"] = p["review"].pop("original_text")
                continue
        p["review"].update(verdict="rejected" if a == "r" else "approved", reviewer=reviewer, on=date.today().isoformat())
        if a == "r":
            p["review"]["note"] = input_fn("why (optional) > ").strip()
    if ledger["review"].get("label") != "approved" and _confirm_label(ledger, reviewer, input_fn, print_fn) == "q":
        save()
        return ledger
    live = [p for p in ledger["propositions"] if p["review"]["verdict"] != "rejected"]
    if ledger.get("ground_truth") and any(p["id"] in ledger["ground_truth"]["uses"] and p["review"]["verdict"] == "rejected" for p in ledger["propositions"]):
        print_fn("\nThe ground-truth answer relies on a rejected proposition. Edit `ground_truth` in ledger.json (or rebuild with better sources), then re-run `check`.")
    elif ledger.get("ground_truth") and all(p["review"]["verdict"] == "approved" for p in live if p["id"] in ledger["ground_truth"]["uses"]):
        print_fn(f"\nGROUND TRUTH: {ledger['ground_truth']['answer']}\nBuilt from {', '.join(ledger['ground_truth']['uses'])}. Does it say only what those approved claims say?")
        if _ask(input_fn, "[a]pprove  [r]eject  [s]kip > ", "ars") != "s":
            ledger["review"]["answer"] = "approved" if _ask(input_fn, "confirm approve? [a]=yes [r]=no > ", "ar") == "a" else "rejected"
    for f in gates.FRAMINGS:
        if any(it["framing"] == f and it["review"].get("verdict") == "approved" for it in ledger["items"]):
            continue
        if _choose(ledger, f, reviewer, "\n".join(docs.values()), input_fn, print_fn) == "q":
            break
    save()
    write_review_md(ledger, docs, run_dir / "review.md")
    return ledger
