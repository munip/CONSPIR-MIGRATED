"""Guardrails that run before work is spent (doctor, topic overlap, source-set warnings) and a progress report."""
from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import gates

REPO = Path(__file__).resolve().parent.parent.parent
_STOP = {"theory", "theories", "conspiracy", "claim", "claims", "about", "what", "that", "this", "with", "from", "were",
         "have", "death", "crash", "event", "story", "case", "official", "truth", "real", "really", "did", "does"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{4,}", text.casefold()) if w not in _STOP}


def topic_overlap(topic: str, repo: Path = REPO, threshold: float = 0.5) -> list[dict]:
    """Existing questions/claim_ids that share most of the topic's words. Advisory: open PRs are not visible here."""
    want = _words(topic)
    if not want:
        return []
    hits = []
    for path in sorted((repo / "questions").rglob("*.jsonl")):
        if path.name == "template.jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            have = _words(rec.get("claim_id", "").replace("-", " ") + " " + rec.get("question", ""))
            score = len(want & have) / len(want)
            if score >= threshold:
                hits.append({"claim_id": rec.get("claim_id", ""), "id": rec.get("id", ""), "question": rec.get("question", ""),
                             "file": path.name, "score": round(score, 2)})
    return sorted(hits, key=lambda h: -h["score"])[:5]


def source_warnings(specs: list[str]) -> list[str]:
    uris = [s.partition(" | ")[0].strip() for s in specs]
    hosts = {urllib.parse.urlparse(u).hostname or "local file" for u in uris}
    out = []
    if len(uris) == 1:
        out.append("only one source: add a second, independent source if you can, so each fact is corroborated")
    elif len(hosts) == 1:
        out.append(f"all sources come from one host ({next(iter(hosts))}): they may not be independent")
    return out


def doctor(base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b", repo: Path = REPO, topic: str | None = None,
           fetch_tags=None) -> tuple[list[str], bool]:
    """Return (report lines, all_ok)."""
    lines, ok = [], True

    def check(good: bool, label: str, fix: str = ""):
        nonlocal ok
        ok &= good
        lines.append(("  ok    " if good else "  FIX   ") + label + ("" if good else f"\n          -> {fix}"))
    check(sys.version_info >= (3, 9), f"Python {sys.version.split()[0]}", "use Python 3.9 or newer")
    try:
        import jsonschema  # noqa: F401
        check(True, "jsonschema installed")
    except ImportError:
        check(False, "jsonschema installed", "python3 -m venv .venv && .venv/bin/pip install jsonschema, then run with .venv/bin/python")
    check((repo / "schema.json").is_file() and (repo / "questions").is_dir(), "run from the repository root (schema.json and questions/ found)",
          "cd into your ConspiracyBench clone")
    host = urllib.parse.urlparse(base_url).hostname
    check(host in ("localhost", "127.0.0.1", "::1"), f"model server is local ({base_url})", "use a local server, or pass --allow-remote-llm knowingly")
    try:
        tags = fetch_tags() if fetch_tags else json.loads(urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=5).read())
        names = [m["name"] for m in tags.get("models", [])]
        check(True, f"model server reachable ({len(names)} models)")
        check(any(n == model or n.startswith(model + ":") or n.split(":")[0] == model for n in names), f"model {model!r} is pulled",
              f"ollama pull {model}   (installed: {', '.join(names) or 'none'})")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        check(False, "model server reachable", f"start it (`ollama serve`, or open the Ollama app): {exc}")
    if topic:
        hits = topic_overlap(topic, repo)
        lines.append(f"  topic overlap for {topic!r}: " + ("none found in questions/ (also search open PRs and discussions on the Space)" if not hits else ""))
        lines += [f"    {h['claim_id'] or h['id']} ({h['file']}, {h['score']}): {h['question'][:90]}" for h in hits]
        if hits:
            lines.append("    -> same event? Only continue if you are adding a genuinely different angle or framing: set the existing claim_id at the\n"
                         "       label step and export with `finalize --extend-claim`. Otherwise choose another topic.")
    return lines, ok


def progress(ledger: dict) -> tuple[list[tuple[bool, str]], str]:
    """Checklist of where a run stands, and the next command to run."""
    props = ledger["propositions"]
    verdicts = [p["review"].get("verdict") for p in props]
    gt = ledger.get("ground_truth")
    uses = set(gt["uses"]) if gt else set()
    cited_ok = bool(gt) and all(p["review"].get("verdict") == "approved" for p in props if p["id"] in uses)
    chosen = {it["framing"] for it in ledger.get("items", []) if it["review"].get("verdict") == "approved"}
    rev = ledger["review"]
    steps = [
        (bool(gt), "draft built (claim label, answer, candidates)"),
        (cited_ok, f"claims the answer relies on approved ({sum(v == 'approved' for v in verdicts)} approved, {sum(v == 'rejected' for v in verdicts)} rejected, {sum(v is None for v in verdicts)} undecided)"),
        (rev.get("label") == "approved", "claim label confirmed (claim_id, category, region, allegation)"),
        (rev.get("answer") == "approved", "ground-truth answer approved"),
        *[(f in chosen, f"{f} candidate chosen") for f in gates.FRAMINGS],
    ]
    todo = next((i for i, (done, _) in enumerate(steps) if not done), None)
    return steps, ("finalize" if todo is None else "review")
