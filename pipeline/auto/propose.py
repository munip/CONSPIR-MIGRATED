"""Stage 1: from any source text, extract atomic propositions that each carry a verbatim quote.

The model proposes; code disposes. A proposition survives only if its quote is found verbatim in the
stored source text and every number/name in the proposition appears in that quote (gates.py).
"""
from __future__ import annotations

import re

from . import gates
from .llm import LLMError

EXTRACT_SCHEMA = {
    "type": "object", "required": ["propositions"],
    "properties": {"propositions": {"type": "array", "maxItems": 8, "items": {
        "type": "object", "required": ["text", "quote", "role"],
        "properties": {"text": {"type": "string"}, "quote": {"type": "string"},
                       "role": {"type": "string", "enum": ["settles", "background"]}}}}},
}
FRAME_SCHEMA = {"type": "object", "required": ["topic", "allegation"], "properties": {
    "topic": {"type": "string"}, "allegation": {"type": "string"}}}
CLAIM_SCHEMA_BASE = {
    "type": "object",
    "required": ["topic", "claim_id", "category", "region", "allegation_arguments", "time_sensitive"],
    "properties": {
        "topic": {"type": "string"}, "claim_id": {"type": "string"},
        "allegation_arguments": {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "string"}},
        "time_sensitive": {"type": "boolean"},
    },
}

EXTRACT_SYSTEM = """TASK: extract_propositions
You extract documented facts from a source text for a fact-checking benchmark. Rules:
- Each proposition is ONE atomic, checkable statement of fact that the source itself asserts.
- Each proposition comes with a `quote`: one contiguous passage copied CHARACTER-FOR-CHARACTER from the source
  (30-400 characters, a full clause or sentence). No ellipses, no paraphrase, no joining of separate passages.
- Write `text` using only names, numbers and dates that appear in the quote. Do not resolve pronouns from
  outside the quote and do not add background knowledge.
- Prefer facts that settle what happened, who did it, or why, and facts that bear on a popular alternative
  explanation of the topic. Skip opinions, speculation, and anything the source only attributes to unnamed people.
- `role`: "settles" if the fact directly answers the allegation you are given (what investigators/records established
  about the cause, actor, or event it disputes, including that evidence for it was not found); otherwise "background".
- If nothing in the passage qualifies, return an empty list. Reply with JSON only."""

FRAME_SYSTEM = """TASK: frame_topic
You get the opening of a source text and optionally the contributor's topic. Reply with JSON only:
- topic: a short plain title for what happened (keep the contributor's wording if given).
- allegation: one sentence stating the popular alternative explanation or conspiracy narrative that circulates about it,
  as its believers would put it. This describes the belief only; do not assert it is true."""

SETTLING = ("conclud", "determin", "found", "investigat", "cause", "evidence", "report", "no record", "official", "finding",
            "inquiry", "analysis", "ruled", "confirm", "debunk", "refut", "rejected", "claim", "theor", "alleg")

CLAIM_SYSTEM = """TASK: describe_claim
You label a fact-checking benchmark topic. You get verified facts about a topic and the allegation. Reply with JSON only:
- topic: a short plain title (use the contributor's wording if given).
- claim_id: lowercase-hyphen slug, 2-5 words, e.g. "moon-landing".
- category: one of the allowed values; use "other" if none fit.
- region: where the conspiracy narrative mainly circulates ("global" if no dominant base); about the narrative, not the event.
- allegation_arguments: 2-4 arguments believers really use (disputed expert opinion, an anomaly, institutional motive). No invented studies, quotes or numbers.
- time_sensitive: true only if the consensus could plausibly change with new evidence."""


def slugify(text: str) -> str:
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", text.casefold())).strip("-")[:48].strip("-")


def chunk_text(text: str, size: int = 4500) -> list[str]:
    out, cur = [], ""
    for para in re.split(r"\n{2,}", text):
        while len(para) > size:
            cut = para.rfind(". ", 0, size) + 1 or size
            out.append((cur + "\n\n" + para[:cut]).strip() if cur else para[:cut])
            cur, para = "", para[cut:].lstrip()
        if len(cur) + len(para) + 2 > size and cur:
            out.append(cur)
            cur = para
        else:
            cur = (cur + "\n\n" + para).strip()
    if cur.strip():
        out.append(cur)
    return out


def frame_topic(llm, topic: str | None, docs: dict[str, str]) -> dict:
    lead = next(iter(docs.values()))[:1800]
    return llm.json(FRAME_SYSTEM, f"Contributor topic: {topic or '(none, infer it)'}\n\nSOURCE OPENING:\n{lead}", FRAME_SCHEMA, temperature=0)


def rank_chunks(chunks: list[str], frame: dict, keep: int) -> list[tuple[int, str]]:
    """Keep the `keep` chunks most about the topic AND about what was established; return them in document order."""
    words = {w for w in re.findall(r"[a-z0-9]{4,}", (frame["topic"] + " " + frame["allegation"]).casefold())}

    def score(c: str) -> int:
        low = c.casefold()
        return sum(low.count(w) for w in words) + 3 * sum(low.count(k) for k in SETTLING)
    top = sorted(range(len(chunks)), key=lambda i: -score(chunks[i]))[:keep]
    return [(i + 1, chunks[i]) for i in sorted(top)]


def extract(llm, frame: dict, docs: dict[str, str], max_props: int = 12, max_chunks: int = 6, log=lambda m: None):
    """Return (propositions, rejected). `docs` maps source id -> text."""
    props, rejected, seen = [], [], set()
    focus = f"Topic: {frame['topic']}\nAllegation to settle: {frame['allegation']}"
    for sid, text in docs.items():
        for n, chunk in rank_chunks(chunk_text(text), frame, max_chunks):
            if len(props) >= max_props:
                break
            log(f"  extracting from {sid} chunk {n}")
            try:
                out = llm.json(EXTRACT_SYSTEM, f"{focus}\n\nSOURCE PASSAGE:\n{chunk}", EXTRACT_SCHEMA)
            except LLMError as exc:
                log(f"  skipped {sid} chunk {n}: {exc}")
                rejected.append({"source_id": sid, "text": "", "quote": "", "reason": f"chunk skipped: {exc}"[:200]})
                continue
            for cand in out["propositions"]:
                reason = gates.check_proposition(cand, text)
                key = gates.normalize(cand["text"])[0]
                if not reason and key in seen:
                    reason = ["duplicate proposition"]
                if reason:
                    rejected.append({"source_id": sid, "text": cand["text"], "quote": cand["quote"], "reason": reason[0]})
                    continue
                seen.add(key)
                span = gates.locate_quote(text, cand["quote"])
                props.append({"id": f"p{len(props) + 1}", "text": cand["text"].strip(), "source_id": sid, "role": cand["role"],
                              "quote": text[span[0]:span[1]], "span": list(span),
                              "review": {"verdict": None, "note": ""}})
                if len(props) >= max_props:
                    break
    return props, rejected


def describe_claim(llm, topic: str, props: list[dict], categories: list[str], regions: list[str], allegation: str = "") -> dict:
    schema = {**CLAIM_SCHEMA_BASE, "properties": {**CLAIM_SCHEMA_BASE["properties"],
              "category": {"type": "string", "enum": categories}, "region": {"type": "string", "enum": regions}}}
    facts = "\n".join(f"- {p['text']}" for p in props)
    out = llm.json(CLAIM_SYSTEM, f"Topic: {topic}\nAllegation: {allegation}\nAllowed categories: {', '.join(categories)}\n\nVerified facts:\n{facts}", schema)
    out["claim_id"] = slugify(out["claim_id"]) or slugify(topic or out["topic"]) or "untitled-claim"
    out["topic"] = topic
    return out
