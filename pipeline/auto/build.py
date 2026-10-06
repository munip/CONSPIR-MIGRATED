"""Orchestrate: sources -> propositions -> claim label -> ground truth -> three multi-turn framings -> ledger."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from . import gates, generate, guard, propose, sources
from .review import write_review_md

class BuildError(RuntimeError):
    pass


REPO = Path(__file__).resolve().parent.parent.parent
MIN_PROPS = 2


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_run(llm, specs: list[str], out_root: Path, *, topic: str | None = None, schema: dict | None = None,
              max_props: int = 12, candidates: int = 3, audit: bool = True, overwrite: bool = False, user_agent: str = sources.USER_AGENT, refresh: bool = False,
              fetcher=None, cache_dir: Path | None = None, log=lambda m: print(m, file=sys.stderr)) -> tuple[Path, dict, dict]:
    """Returns (run_dir, ledger, report). Raises sources.SourceError / llm.LLMError on hard failures."""
    schema = schema or gates.load_json(REPO / "schema.json")
    cache = Path(cache_dir) if cache_dir else REPO / "pipeline" / "cache" / "auto"
    docs, metas = {}, []
    for n, spec in enumerate(specs, 1):
        kw = {"fetcher": fetcher} if fetcher else {}
        rec = sources.load_source(spec, cache_dir=cache, user_agent=user_agent, refresh=refresh, **kw)
        sid = f"s{n}"
        docs[sid] = rec["text"]
        metas.append({"id": sid, **{k: rec[k] for k in ("uri", "kind", "citation", "title", "sha256", "retrieved_on")}})
        log(f"loaded {sid}: {rec['uri']} ({len(rec['text'])} chars)")
    k = candidates
    report: dict = {"model": llm.name, "date": date.today().isoformat(), "rejected_propositions": [], "warnings": []}
    report["warnings"] += guard.source_warnings(specs)
    for m in metas:
        if sources.is_wikimedia(m["uri"]):
            report["warnings"].append(f"{m['id']} is a Wikimedia page: tertiary, so cite the primary sources it points to when you can")
        if m["kind"] == "file" and not m["citation"]:
            report["warnings"].append(f"{m['id']} is a local file with no citation; finalize needs `path | citation` or a public URL")

    log("stage 0: framing topic and allegation")
    frame = propose.frame_topic(llm, topic, docs)
    if topic:
        frame["topic"] = topic
    log(f"  topic: {frame['topic']} | allegation: {frame['allegation']}")
    run_slug = propose.slugify(frame["topic"]) or "run"
    if (Path(out_root) / run_slug / "ledger.json").exists() and not overwrite:
        raise BuildError(f"{Path(out_root) / run_slug} already holds a draft (and maybe your review progress). Continue it with `review`, "
                         "or pass --overwrite to discard it and start again")
    for h in guard.topic_overlap(frame["topic"]):
        report["warnings"].append(f"possible duplicate of existing {h['claim_id'] or h['id']} ({h['file']}): {h['question'][:80]}")
    log("stage 1: extracting propositions")
    props, rejected = propose.extract(llm, frame, docs, max_props=max_props, log=log)
    report["rejected_propositions"] = rejected
    log(f"  kept {len(props)}, rejected {len(rejected)}")
    ledger = {"version": 2, "status": "needs_human_verification", "topic": frame["topic"], "sources": metas, "propositions": props,
              "language": "en", "built_with": {"model": llm.name, "date": date.today().isoformat()}, "items": [],
              "review": {"reviewer": None, "answer": None}}
    slug = run_slug
    if len(props) < MIN_PROPS:
        report["status"] = f"failed: only {len(props)} verified proposition(s); need {MIN_PROPS}. Add richer sources or name a --topic."
        return _save(out_root, slug, ledger, report, docs)

    log("stage 2: labelling claim")
    cats, regions = schema["properties"]["category"]["enum"], schema["properties"]["region"]["enum"]
    if not any(p["role"] == "settles" for p in props):
        report["status"] = ("failed: no extracted proposition settles the allegation (sources describe the event but not what was established). "
                            "Add a source that states the investigation's/record's conclusion.")
        return _save(out_root, slug, ledger, report, docs)
    meta = propose.describe_claim(llm, frame["topic"], props, cats, regions, frame["allegation"])
    ledger.update({k: meta[k] for k in ("claim_id", "category", "region", "allegation_arguments")})
    ledger["alternative_allegation"] = frame["allegation"]
    if meta["time_sensitive"]:
        ledger["consensus_as_of"] = date.today().isoformat()

    log("stage 3: ground-truth answer")
    answer, problems = generate.write_answer(llm, ledger)
    if not answer:
        report["status"] = "failed: no grounded ground-truth answer: " + "; ".join(problems)
        return _save(out_root, slug, ledger, report, docs)
    ledger["ground_truth"] = answer

    all_text = "\n".join(docs.values())
    ledger["candidates"] = {}
    for framing in gates.FRAMINGS:
        log(f"stage 4: drafting {k} {framing} candidates (multi-turn)")
        cands = generate.draft_candidates(llm, ledger, framing, k=k, audit=audit, docs_text=all_text, log=log)
        ledger["candidates"][framing] = cands
        usable = sum(1 for c in cands if not c["blocking"])
        log(f"  {len(cands)} drafted, {usable} pass the deterministic gates")
    stuck = [f for f in gates.FRAMINGS if not any(not c["blocking"] for c in ledger["candidates"][f])]
    report["status"] = "ok" if not stuck else f"needs_edit: no candidate for {stuck} passes the gates as drafted; you can edit one in `review` or write it in ledger.json"
    return _save(out_root, slug, ledger, report, docs, schema)


def _save(out_root: Path, slug: str, ledger: dict, report: dict, docs: dict, schema: dict | None = None):
    run_dir = Path(out_root) / slug
    for sid, text in docs.items():
        _write(run_dir / "docs" / f"{sid}.txt", text)
    if schema is not None and ledger.get("propositions"):
        problems = gates.check_ledger(ledger, run_dir, None, drafting=True)
        if problems:
            report["status"] = "failed: ledger gates: " + "; ".join(problems)
    _write(run_dir / "ledger.json", json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
    _write(run_dir / "report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_review_md(ledger, docs, run_dir / "review.md", report)
    return run_dir, ledger, report
