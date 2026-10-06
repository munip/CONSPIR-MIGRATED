"""Step 1: discover candidate topics from Wikipedia's "List of conspiracy theories".

Draft-only. Each candidate points at a Wikipedia ARTICLE whose own cited references
become candidate sources in the next step (harvest_references.py). The `claim` text is
context copied from the list page to orient a human reviewer; it is NOT verified and
Wikipedia is never used as a ground-truth citation.

Run from the repo root:
    python -m pipeline.discover_topics --out pipeline/out/candidate_topics.jsonl
    python -m pipeline.discover_topics --list-html saved_page.html --out ...
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
from collections import Counter
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

from . import wiki
from .io_utils import DRAFT_STATUS, OutputError, make_console_safe, write_jsonl

SKIP_SECTIONS = {
    "see also", "references", "notes", "footnotes", "further reading",
    "external links", "bibliography", "sources", "citations", "works cited",
}
GENERIC_TITLES = {
    "conspiracy theory", "conspiracy", "conspiracy theories", "conspiracy theorist",
    "conspiracy theorists", "false flag", "disinformation", "misinformation",
    "fake news", "hoax",
}
# Strong title cues are enough on their own. Weak cues (event-style titles and every
# "Main article" hatnote) are kept only if the surrounding sentence talks about
# conspiracy/allegation, which drops unrelated links such as "Death of Jesus Christ".
STRONG_RE = re.compile(
    r"conspirac|hoax|cover[- ]?up|denial|false flag|theor(?:y|ies)|sightings|controvers",
    re.IGNORECASE,
)
WEAK_RE = re.compile(
    r"incident|\b(?:death|assassination|murder|disappearance|suicide) of\b", re.IGNORECASE
)
CONTEXT_RE = re.compile(
    r"conspir|theor|alleg|claim|suspic|cover[- ]?up|foul play|unsolved|mysteri|hoax|rumou?r",
    re.IGNORECASE,
)
MAX_LINKS_PER_CLAIM = 5  # more internal links than this in one sentence = an enumeration
_ABBREV_END = re.compile(r"(?:\b[A-Z]|\b(?:Jr|Sr|St|Dr|Mr|Mrs|Ms|vs|No|Inc|Ltd|Co)|U\.S)\.$")
_SKIP_TABLE = ("infobox", "navbox", "sidebar", "metadata", "ambox", "vertical-navbox")
_SKIP_DIV = ("navbox", "sidebar", "thumb", "reflist", "refbegin", "toc", "metadata",
             "mw-references-wrap")
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"“(])')


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends but not after initials/abbreviations ("George S. Patton")."""
    merged: list[str] = []
    for part in _SENTENCE_SPLIT.split(text):
        if merged and _ABBREV_END.search(merged[-1]):
            merged[-1] += " " + part
        else:
            merged.append(part)
    return merged


def _has_class(classes: list[str], prefixes) -> bool:
    return any(c.startswith(p) for c in classes for p in prefixes)


class _ListPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict] = []
        self._heads: dict[int, str] = {}
        self._heading_level: int | None = None
        self._heading_buf: list[str] = []
        self._skip: tuple[str, int] | None = None
        self._block: dict | None = None
        self._hat_depth = 0
        self._link: dict | None = None

    # -- helpers
    def _section(self) -> str:
        return " > ".join(self._heads[k] for k in sorted(self._heads)) or "(lead)"

    def _in_skipped_section(self) -> bool:
        return any(h.lower() in SKIP_SECTIONS for h in self._heads.values())

    def _flush(self) -> None:
        block, self._block = self._block, None
        if block and not self._in_skipped_section():
            block["text"] = _clean("".join(block["text"]))
            block["section"] = self._section()
            self.blocks.append(block)

    def _begin_skip(self, tag: str) -> None:
        self._skip = (tag, 1)

    # -- HTMLParser hooks
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if self._skip:
            if tag == self._skip[0]:
                self._skip = (tag, self._skip[1] + 1)
            return
        if tag in ("style", "script", "nav", "footer", "header"):
            return self._begin_skip(tag)
        if tag == "sup" and ("reference" in classes or "noprint" in classes):
            return self._begin_skip(tag)
        if tag == "span" and _has_class(classes, ("mw-editsection",)):
            return self._begin_skip(tag)
        if tag == "table" and _has_class(classes, _SKIP_TABLE):
            return self._begin_skip(tag)
        if tag == "div":
            if self._hat_depth:
                self._hat_depth += 1
                return
            if "hatnote" in classes:
                self._flush()
                self._hat_depth = 1
                self._block = {"kind": "hatnote", "text": [], "links": []}
                return
            if _has_class(classes, _SKIP_DIV):
                return self._begin_skip(tag)
            return
        if tag in ("h2", "h3", "h4"):
            self._flush()
            self._heading_level = int(tag[1])
            self._heading_buf = []
            return
        if tag in ("p", "li"):
            self._flush()
            self._block = {"kind": "inline", "text": [], "links": []}
            return
        if tag == "a" and self._block is not None and self._heading_level is None:
            self._link = {"href": a.get("href", ""), "buf": []}
        elif tag == "br" and self._block is not None:
            self._block["text"].append(" ")

    def handle_endtag(self, tag):
        if self._skip:
            if tag == self._skip[0]:
                depth = self._skip[1] - 1
                self._skip = (tag, depth) if depth > 0 else None
            return
        if tag in ("h2", "h3", "h4") and self._heading_level == int(tag[1]):
            level, text = self._heading_level, _clean("".join(self._heading_buf))
            self._heads[level] = text
            for deeper in range(level + 1, 5):
                self._heads.pop(deeper, None)
            self._heading_level = None
        elif tag == "a" and self._link is not None:
            if self._block is not None:
                self._block["links"].append((self._link["href"], _clean("".join(self._link["buf"]))))
            self._link = None
        elif tag in ("p", "li"):
            self._flush()
        elif tag == "div" and self._hat_depth:
            self._hat_depth -= 1
            if self._hat_depth == 0:
                self._flush()

    def handle_data(self, data):
        if self._skip:
            return
        if self._heading_level is not None:
            self._heading_buf.append(data)
            return
        if self._link is not None:
            self._link["buf"].append(data)
        if self._block is not None:
            self._block["text"].append(data)

    def close(self):
        super().close()
        self._flush()


def wiki_title(href: str) -> str | None:
    """Article title for an internal Wikipedia link, or None (namespaces, externals)."""
    if not href:
        return None
    for prefix in ("/wiki/", "./", "https://en.wikipedia.org/wiki/"):
        if href.startswith(prefix):
            rest = href[len(prefix):]
            break
    else:
        return None
    rest = rest.split("#", 1)[0].split("?", 1)[0]
    title = urllib.parse.unquote(rest).replace("_", " ").strip()
    return None if (not title or ":" in title) else title


def title_strength(title: str, kind: str) -> str | None:
    lowered = title.lower()
    if lowered.startswith("list of") or lowered in GENERIC_TITLES:
        return None
    if STRONG_RE.search(title):
        return "strong"
    if kind == "hatnote" or WEAK_RE.search(title):
        return "weak"
    return None


def _shorten(text: str, limit: int = 400) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def parse_list_page(html: str, *, page_title: str = wiki.LIST_PAGE_TITLE,
                    revid: int | None = None, retrieved_on: str | None = None) -> list[dict]:
    parser = _ListPageParser()
    parser.feed(html)
    parser.close()
    retrieved_on = retrieved_on or date.today().isoformat()
    found: dict[str, dict] = {}
    for index, block in enumerate(parser.blocks):
        # A hatnote's claim comes only from the paragraph right after it (skipping stacked hatnotes).
        j = index + 1
        while j < len(parser.blocks) and parser.blocks[j]["kind"] == "hatnote":
            j += 1
        following = (parser.blocks[j] if j < len(parser.blocks)
                     and parser.blocks[j]["section"] == block["section"] else None)
        sentences = split_sentences(block["text"]) or [block["text"]]
        for href, link_text in block["links"]:
            title = wiki_title(href)
            strength = title_strength(title, block["kind"]) if title else None
            if strength is None:
                continue
            origin = "wikipedia_list_context"
            if block["kind"] == "hatnote":
                claim = _shorten(" ".join(split_sentences(following["text"])[:2])) if following else ""
                context = claim
            else:
                sentence = next((s for s in sentences if link_text and link_text in s), sentences[0])
                context = sentence
                n_links = sum(1 for h, t in block["links"] if t and t in sentence and wiki_title(h))
                if n_links > MAX_LINKS_PER_CLAIM:
                    claim, origin = "", "enumeration_sentence"
                else:
                    claim = _shorten(sentence)
            if strength == "weak" and not CONTEXT_RE.search(context):
                continue
            if not claim and origin == "wikipedia_list_context":
                origin = "none"
            if title in found:
                found[title]["mentions"] += 1
                continue
            found[title] = {
                "topic": title,
                "claim": claim,
                "claim_origin": origin,
                "source_article_url": wiki.article_url(title),
                "list_section": block["section"],
                "link_origin": block["kind"],
                "mentions": 1,
                "status": DRAFT_STATUS,
                "discovered_from": {"page": page_title, "revid": revid, "retrieved_on": retrieved_on},
            }
    return list(found.values())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("pipeline/out/candidate_topics.jsonl"))
    p.add_argument("--list-html", type=Path, help="Use a saved HTML file instead of the network")
    p.add_argument("--cache-dir", type=Path, default=Path("pipeline/cache"))
    p.add_argument("--user-agent", default=wiki.DEFAULT_USER_AGENT)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--refresh", action="store_true", help="Ignore the on-disk cache")
    p.add_argument("--limit", type=int, help="Keep only the first N candidates")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    make_console_safe()
    try:
        if args.list_html:
            html, revid, retrieved = args.list_html.read_text(encoding="utf-8"), None, None
        else:
            rec = wiki.fetch_page(wiki.LIST_PAGE_TITLE, cache_dir=args.cache_dir,
                                  user_agent=args.user_agent, delay=args.delay, refresh=args.refresh)
            html, revid, retrieved = rec["html"], rec["revid"], rec["retrieved_on"]
        candidates = parse_list_page(html, revid=revid, retrieved_on=retrieved)
        if args.limit:
            candidates = candidates[: args.limit]
        write_jsonl(candidates, args.out, overwrite=args.overwrite)
    except (wiki.FetchError, OutputError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    sections = Counter(c["list_section"].split(" > ")[0] for c in candidates)
    print(f"Wrote {len(candidates)} draft candidates to {args.out} (all {DRAFT_STATUS}).")
    for name, n in sections.most_common(10):
        print(f"  {n:3d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
