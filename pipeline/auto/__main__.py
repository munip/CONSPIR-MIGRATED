"""python -m pipeline.auto {build,review,check,finalize} ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..io_utils import make_console_safe
from . import finalize as fin, gates, guard, llm as llmmod, sources
from .build import REPO, BuildError, build_run
from .review import interactive_review

OUT = REPO / "pipeline" / "out" / "auto"


def main(argv=None) -> int:
    make_console_safe()
    ap = argparse.ArgumentParser(prog="python -m pipeline.auto", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="sources -> draft ledger (needs a local model)")
    b.add_argument("--topic", help="what the sources are about, in your words (recommended)")
    b.add_argument("--source", action="append", default=[], metavar="URL|PATH[ | citation]", help="repeatable; any URL, or a local .txt/.html file")
    b.add_argument("--sources-file", type=Path, help="text file, one source per line")
    b.add_argument("--from-candidates", type=Path, help="candidates_with_sources.jsonl from harvest_references; uses --topic's sources")
    b.add_argument("--backend", default="ollama", choices=["ollama", "openai"])
    b.add_argument("--model", default=llmmod.DEFAULT_MODEL)
    b.add_argument("--base-url")
    b.add_argument("--allow-remote-llm", action="store_true")
    b.add_argument("--max-props", type=int, default=12)
    b.add_argument("--candidates", type=int, default=3, help="drafts per framing for the reviewer to choose from (default 3)")
    b.add_argument("--no-audit", action="store_true", help="skip the model-based framing/turn audits (not recommended)")
    b.add_argument("--overwrite", action="store_true", help="discard an existing draft (and its review progress) for the same topic")
    b.add_argument("--refresh", action="store_true", help="re-download cached URLs")
    b.add_argument("--user-agent", default=sources.USER_AGENT)
    b.add_argument("--out", type=Path, default=OUT)
    d = sub.add_parser("doctor", help="check your setup (and optionally a topic) before you start")
    d.add_argument("--topic")
    d.add_argument("--model", default=llmmod.DEFAULT_MODEL)
    d.add_argument("--base-url", default="http://localhost:11434")
    st = sub.add_parser("status", help="show what is done and what is left for a run")
    st.add_argument("run_dir", type=Path)
    r = sub.add_parser("review", help="interactively verify propositions, answer and items")
    r.add_argument("run_dir", type=Path)
    r.add_argument("--reviewer", required=True, help="your name / HF username")
    c = sub.add_parser("check", help="re-run all deterministic gates (after hand-editing ledger.json)")
    c.add_argument("run_dir", type=Path)
    f = sub.add_parser("finalize", help="export approved items to <run>/final/ (never writes questions/)")
    f.add_argument("run_dir", type=Path)
    f.add_argument("--contributor", help="HF username for the contributor field (default: the reviewer)")
    f.add_argument("--extend-claim", action="store_true")
    f.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "build":
            specs = list(a.source)
            if a.sources_file:
                specs += [ln.strip() for ln in a.sources_file.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
            if a.from_candidates:
                if not a.topic:
                    ap.error("--from-candidates needs --topic")
                specs += sources.sources_from_candidates(a.from_candidates, a.topic)
            if not specs:
                ap.error("give at least one --source")
            model = llmmod.make_llm(a.backend, a.model, a.base_url, a.allow_remote_llm)
            run_dir, ledger, report = build_run(model, specs, a.out, topic=a.topic, max_props=a.max_props, candidates=a.candidates, audit=not a.no_audit,
                                                user_agent=a.user_agent, refresh=a.refresh)
            print(f"\n{report['status']}\nrun dir: {run_dir}\nnext: python -m pipeline.auto review {run_dir} --reviewer <you>   (read {run_dir / 'review.md'} first)")
            return 0 if report["status"] == "ok" else 2
        if a.cmd == "doctor":
            lines, ok = guard.doctor(a.base_url, a.model, REPO, a.topic)
            print("\n".join(lines) + ("\n\nready." if ok else "\n\nfix the items marked FIX, then run doctor again."))
            return 0 if ok else 1
        if a.cmd == "status":
            ledger = gates.load_json(a.run_dir / "ledger.json")
            steps, nxt = guard.progress(ledger)
            print(f"{ledger['topic']} ({ledger['claim_id']})")
            print("\n".join(("  [x] " if done else "  [ ] ") + label for done, label in steps))
            print(f"\nnext: python -m pipeline.auto {nxt} {a.run_dir}" + ("" if nxt == "finalize" else " --reviewer <you>"))
            return 0
        if a.cmd == "review":
            interactive_review(a.run_dir, a.reviewer)
            print(f"saved. next: python -m pipeline.auto finalize {a.run_dir}")
            return 0
        if a.cmd == "check":
            ledger = gates.load_json(a.run_dir / "ledger.json")
            problems = gates.check_ledger(ledger, a.run_dir, gates.load_json(REPO / "schema.json"), drafting=not ledger.get("items"))
            print("\n".join(problems) if problems else ("all gates pass" if ledger.get("items") else "sources, propositions and answer pass; no items chosen yet (run review)"))
            return 1 if problems else 0
        paths = fin.finalize(a.run_dir, a.contributor, extend_claim=a.extend_claim, overwrite=a.overwrite)
        print("wrote:\n  " + "\n  ".join(map(str, paths)) +
              f"\n\nnext (you, not the tool):\n  cp {paths[0]} {REPO / 'questions' / 'pending'}/\n"
              "  python scripts/validate_questions.py && python scripts/check_duplicates.py\n"
              f"then open ONE pull request with that file, using {paths[2].name} as the PR description (it lists each claim with its quote and source;\n"
              f"  e.g. hf upload ... --create-pr --commit-description \"$(cat {paths[2]})\"). Keep {paths[1].name} in case a maintainer asks to re-check.")
        return 0
    except (sources.SourceError, llmmod.LLMError, fin.FinalizeError, BuildError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
