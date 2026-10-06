"""Export only what a person approved, re-checking every gate against the stored source texts first."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from . import gates

REPO = Path(__file__).resolve().parent.parent.parent


class FinalizeError(ValueError):
    pass


def existing_inventory(repo: Path = REPO) -> tuple[set[str], set[str]]:
    claims, questions = set(), set()
    for path in sorted((repo / "questions").rglob("*.jsonl")):
        if path.name == "template.jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                questions.add(rec["question"].strip().casefold())
                if rec.get("claim_id"):
                    claims.add(rec["claim_id"])
    return claims, questions


def finalize(run_dir: Path, contributor: str | None = None, *, repo: Path = REPO, extend_claim: bool = False,
             overwrite: bool = False) -> list[Path]:
    run_dir = Path(run_dir)
    ledger = gates.load_json(run_dir / "ledger.json")
    schema = gates.load_json(repo / "schema.json")
    reviewer = ledger["review"].get("reviewer")
    contributor = contributor or reviewer
    if not reviewer or not contributor:
        raise FinalizeError("no reviewer recorded: run `review` first (a person must verify before export)")
    problems = gates.check_ledger(ledger, run_dir, schema)
    if problems:
        raise FinalizeError("gates failed:\n  - " + "\n  - ".join(problems))
    todo = [f"proposition {p['id']}" for p in ledger["propositions"] if p["id"] in ledger["ground_truth"]["uses"] and p["review"].get("verdict") != "approved"]
    if ledger["review"].get("label") != "approved":
        todo.append("claim label (claim_id, category, region, allegation)")
    if ledger["review"].get("answer") != "approved":
        todo.append("ground-truth answer")
    todo += [f"{it['framing']} item" for it in ledger["items"] if it["review"].get("verdict") != "approved"]
    if todo:
        raise FinalizeError("not yet approved by a human: " + ", ".join(todo))
    if not all(gates.source_ref(s) for s in gates.used_sources(ledger)):
        raise FinalizeError("a cited local-file source has no citation; rebuild with `path | citation text` or use a public URL")
    claims, questions = existing_inventory(repo)
    if ledger["claim_id"] in claims and not extend_claim:
        raise FinalizeError(f"claim_id {ledger['claim_id']!r} already exists in questions/; pass --extend-claim if you are deliberately adding framings of it")
    dup = [it["framing"] for it in ledger["items"] if it["question"].strip().casefold() in questions]
    if dup:
        raise FinalizeError(f"question text already in the corpus for: {', '.join(dup)}")
    out_dir = run_dir / "final"
    jsonl, prov = out_dir / f"{ledger['claim_id']}.jsonl", out_dir / f"{ledger['claim_id']}.provenance.json"
    if (jsonl.exists() or prov.exists()) and not overwrite:
        raise FinalizeError(f"{jsonl} exists; pass --overwrite")
    rows = [gates.render_item(ledger, next(i for i in ledger["items"] if i["framing"] == f), contributor=contributor, added=date.today().isoformat())
            for f in gates.FRAMINGS]
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8", newline="\n")
    prov.write_text(json.dumps({
        "claim_id": ledger["claim_id"], "reviewer": reviewer, "built_with": ledger["built_with"],
        "sources": [s for s in ledger["sources"] if s in gates.used_sources(ledger)],
        "propositions": [p for p in ledger["propositions"] if p["id"] in ledger["ground_truth"]["uses"]],
        "allegation_unsourced": ledger["alternative_allegation"],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    desc = out_dir / f"{ledger['claim_id']}.pr-description.md"
    desc.write_text(pr_description(ledger, reviewer, rows), encoding="utf-8", newline="\n")
    return [jsonl, prov, desc]


def pr_description(ledger: dict, reviewer: str, rows: list[dict]) -> str:
    """Paste-ready PR text: what it tests, the sources, and every approved claim with its verbatim quote."""
    lines = [f"Adds {len(rows)} items (neutral, leading, adversarial; each multi-turn) for claim `{ledger['claim_id']}`: {ledger['topic']}.",
             "", f"Tests whether a model keeps the documented answer when the question presupposes or insinuates: {ledger['alternative_allegation']}",
             "", "Ground truth: " + ledger["ground_truth"]["answer"], "", "Sources:"]
    lines += [f"- {gates.source_ref(s) or s['uri']} (retrieved {s['retrieved_on']})" for s in gates.used_sources(ledger)]
    lines += ["", "Claims behind the ground truth, each verified by the contributor against its source (verbatim quotes):"]
    for p in ledger["propositions"]:
        if p["id"] in ledger["ground_truth"]["uses"]:
            lines.append(f"- {p['text']}  \n  > {p['quote']}")
    lines += ["", f"Drafted with {ledger['built_with']['model']} using pipeline/auto; every claim, the answer and each item were reviewed by {reviewer}."]
    return "\n".join(lines) + "\n"
